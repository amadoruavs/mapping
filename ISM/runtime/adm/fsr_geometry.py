"""Reject offline frame-generation pairs that cannot share one planar camera."""
import cv2
import numpy as np
from .geometry import project


def midpoint_camera(previous,current,shape):
    h,w=shape[:2];corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    relative=np.linalg.inv(previous)@current
    half=(project(relative,corners)+corners)*.5
    result=cv2.getPerspectiveTransform(half.astype(np.float32),project(current,corners).astype(np.float32))
    if not np.isfinite(result).all() or abs(np.linalg.det(result))<1e-12:
        raise ValueError('Degenerate FSR midpoint camera')
    return result


def check_pair(previous,current,shape):
    """A finite midpoint alone does not establish a valid generated map view."""
    h,w=shape[:2]
    try:
        relative=np.linalg.inv(previous)@current
        grid=np.array([(x,y) for y in np.linspace(0,h-1,9) for x in np.linspace(0,w-1,9)])
        old=project(relative,grid);half=(old+grid)*.5
        middle=midpoint_camera(previous,current,shape)
        error=np.linalg.norm(project(np.linalg.inv(current)@middle,half)-grid,axis=1)
        if not np.isfinite(error).all() or not np.isfinite(old).all():
            raise ValueError('Nonfinite FSR camera projection')
        axis=project(relative,[[w/2,h/2],[w/2,h/2-1]])
        direction=axis[1]-axis[0];rotation=abs(float(np.rad2deg(np.arctan2(direction[0],-direction[1]))))
        motion=float(np.percentile(np.linalg.norm(old-grid,axis=1),95))
        corners=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])
        denominators=corners@middle[2]
        reason=('midpoint crosses horizon' if denominators.min()*denominators.max()<=0 else
                'camera rotation exceeds 15 degrees' if rotation>15 else
                'motion exceeds quarter-frame extent' if motion>.25*max(w,h) else
                'half-motion is not a planar camera within 1 pixel' if float(error.max())>1 else None)
        return dict(accepted=reason is None,reason=reason,max_camera_error_px=float(error.max()),
                    rotation_deg=rotation,motion_p95_px=motion),middle
    except (ValueError,np.linalg.LinAlgError,cv2.error):
        return dict(accepted=False,reason='degenerate frame-generation geometry'),None
