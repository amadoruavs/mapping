import os, sys, time, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parallel_runtime.scheduler import bounded_map,worker_count

class SchedulerTests(unittest.TestCase):
    def test_order_and_reservations(self):
        stats={}
        def job(i):time.sleep((7-i%7)*.001);return i*i
        result=list(bounded_map(job,range(100),workers=4,bytes_per_item=17,budget=51,stats=stats))
        self.assertEqual(result,[i*i for i in range(100)])
        self.assertLessEqual(stats['peak_reserved_bytes'],51);self.assertLessEqual(stats['peak_tasks'],3)
        self.assertEqual(stats['submitted'],stats['released']);self.assertEqual(stats['reserved_at_close'],0)
    def test_oversized_never_executes(self):
        called=[]
        with self.assertRaises(MemoryError):list(bounded_map(called.append,[1],bytes_per_item=101,budget=100))
        self.assertEqual(called,[])
    def test_failure_propagates_and_joins(self):
        done=[];stats={}
        def job(i):
            if i==2:raise RuntimeError('injected')
            time.sleep(.001);done.append(i);return i
        with self.assertRaisesRegex(RuntimeError,'injected'):list(bounded_map(job,range(100),workers=4,bytes_per_item=1,budget=4,stats=stats))
        count=len(done);time.sleep(.02);self.assertEqual(len(done),count);self.assertLess(stats['submitted'],10)
    def test_early_close_joins(self):
        stats={};it=bounded_map(lambda x:x,range(10000),workers=4,budget=4,stats=stats)
        next(it);it.close();self.assertLessEqual(stats['submitted'],4);self.assertEqual(stats['reserved_at_close'],0)
    def test_variable_cost(self):
        stats={};values=[3,7,8,2,4]*20
        self.assertEqual(list(bounded_map(lambda x:x,values,workers=4,bytes_per_item=lambda x:x,budget=10,stats=stats)),values)
        self.assertLessEqual(stats['peak_reserved_bytes'],10)
    def test_override_cannot_exceed_stage_budget(self):
        with patch.dict(os.environ,{'ADM_PARALLEL_WORKERS':'256','ADM_PARALLEL':'1'}):
            self.assertEqual(worker_count(2),min(2,os.cpu_count() or 1))
            self.assertEqual(worker_count(1),1)
    def test_serial_switch(self):
        stats={}
        with patch.dict(os.environ,{'ADM_PARALLEL':'0'}):list(bounded_map(lambda x:x,range(10),workers=4,budget=10,stats=stats))
        self.assertEqual(stats['peak_tasks'],1)
if __name__=='__main__':unittest.main()
