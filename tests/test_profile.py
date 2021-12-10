import doctest
import time
import unittest

import fklab.codetools.profile
from fklab.codetools.profile import ExecutionTimer
from fklab.codetools.profile import TimerError


class TestBasicAlgorithm(unittest.TestCase):
    def test_profile(self):
        timer = ExecutionTimer()
        timer.logger = lambda x: None
        timer.start()
        self.assertRaises(TimerError, lambda: timer.start())
        timer.timers = {"test": 0.0}
        timer.name = "test"
        time.sleep(0.01)
        elapsed_time = timer.stop()
        self.assertTrue(isinstance(elapsed_time, float), elapsed_time > 0)
        timer.start()
        time.sleep(0.01)
        elapsed_time_bis = timer.stop()

        self.assertEqual(timer.timers["test"], elapsed_time + elapsed_time_bis)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(fklab.codetools.profile))
    return tests


if __name__ == "__main__":
    unittest.main()
