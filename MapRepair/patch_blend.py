"""Match a geometrically corrected patch to the surrounding map's appearance.

Only intact overlap OUTSIDE every repair supplies the appearance reference.
Color offsets vary slowly; optional softness matching never sharpens a source.
The source homography and map alpha are not altered by this module.
"""
import cv2
import numpy as np


def normalized_blur(values, weight, sigma):
    denominator=cv2.GaussianBlur(weight.astype(np.float32),(0,0),sigma)
    numerator=cv2.GaussianBlur(values*weight[...,None],(0,0),sigma)
    return numerator/np.maximum(denominator[...,None],1e-6),denominator


def match_appearance(source,reference,valid,context,*,match_detail=True):
    """Return float BGR and diagnostics; measurements use valid context only."""
    source=source.astype(np.float32);reference=reference.astype(np.float32)
    valid=valid.astype(np.float32)
    low_s,coverage=normalized_blur(source,valid,6.)
    low_r,_=normalized_blur(reference,valid,6.)
    support=context.astype(bool)&(coverage>.995)&(valid>.999)
    if np.count_nonzero(support)<128:
        return source,dict(context_pixels=int(support.sum()),appearance_matched=False,
                           reason='Too little intact overlap',blur_sigma_px=0.)

    # Median/MAD trim avoids letting a mismatched road edge or moving object
    # dictate the color of the entire patch. All corrections are bounded.
    delta=low_r-low_s;sample=delta[support]
    offset=np.median(sample,axis=0)
    spread=np.maximum(3.,1.4826*np.median(np.abs(sample-offset),axis=0))
    support&=np.all(np.abs(delta-offset)<=2.5*spread,axis=2)
    if np.count_nonzero(support)<128:
        return source,dict(context_pixels=int(support.sum()),appearance_matched=False,
                           reason='Inconsistent appearance reference',blur_sigma_px=0.)
    offset=np.clip(np.median(delta[support],axis=0),-24.,24.)
    # Extrapolate a slow illumination/color field from the ring, falling back
    # smoothly to its robust mean inside a large repaired component.
    field,density=normalized_blur(np.clip(delta,-24.,24.),support.astype(np.float32),24.)
    confidence=np.clip(density/.04,0,1)[...,None]
    field=np.clip(field*confidence+offset*(1-confidence),-24.,24.)

    sigma=0.;before_ratio=1.;after_ratio=1.
    if match_detail:
        # Compare texture in the same intact patches of ground, not the repaired
        # area's possibly shifted edges. Ignore smooth/occluded pixels and cap
        # smoothing at 2 source-map pixels. No deconvolution or detail invention.
        gray_s=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
        gray_r=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
        def energy(gray):
            band=gray-cv2.GaussianBlur(gray,(0,0),1.6)
            return np.sqrt(cv2.GaussianBlur(band*band,(0,0),4.))
        target=energy(gray_r);initial=energy(gray_s)
        usable=support&(target>1.)&(target<30.)&(initial>1.)
        if np.count_nonzero(usable)>=128:
            before_ratio=float(np.median(initial[usable]/target[usable]))
            after_ratio=before_ratio
            if before_ratio>1.15:
                # Retain a small detail advantage instead of forcing exact
                # equality with a soft reference. Pick the least necessary blur.
                best=abs(np.log(before_ratio/1.08))
                for candidate in (.4,.7,1.,1.4,2.):
                    trial=cv2.GaussianBlur(gray_s,(0,0),candidate)
                    ratio=float(np.median(energy(trial)[usable]/target[usable]))
                    loss=abs(np.log(max(ratio,1e-6)/1.08))
                    if loss<best:best=loss;sigma=candidate;after_ratio=ratio
        if sigma:
            source,_=normalized_blur(source,valid,sigma)

    matched=np.clip(source+field,0,255)
    return matched,dict(context_pixels=int(support.sum()),appearance_matched=True,
        color_offset_bgr=offset.tolist(),blur_sigma_px=sigma,
        texture_ratio_before=before_ratio,texture_ratio_after=after_ratio)


def blend_patch(reference,source,selection,source_weight,context,*,feather=32.,match_detail=True):
    """Blend inward, preserving all pixels outside selection exactly.

    reference/source are BGR. source_weight includes source coverage/alpha;
    context identifies intact map pixels, never another repair's reference.
    """
    corrected,report=match_appearance(source,reference,source_weight,context,match_detail=match_detail)
    distance=cv2.distanceTransform(np.pad(selection.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    ramp=np.clip((distance-1)/max(float(feather),1.),0,1)
    weight=ramp*ramp*(3-2*ramp)*source_weight*selection
    # Fine structure gets the narrower transition; the broad color band blends
    # over twice that distance, preventing a low-frequency rectangular halo.
    broad=np.clip((distance-1)/max(float(feather)*2,1.),0,1)
    broad=broad*broad*(3-2*broad)*source_weight*selection
    residual=corrected-reference.astype(np.float32)
    low,_=normalized_blur(residual,(source_weight>.999).astype(np.float32),8.)
    delta=(residual-low)*weight[...,None]+low*broad[...,None]
    result=np.rint(reference.astype(np.float32)+delta).clip(0,255).astype(np.uint8)
    result[weight<=0]=reference[weight<=0]
    report.update(feather_px=float(feather),color_feather_px=float(feather)*2,
                  affected_pixels=int(np.count_nonzero(weight)))
    return result,report
