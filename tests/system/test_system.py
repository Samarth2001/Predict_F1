                                                      

import unittest
import subprocess
import sys
import os

class TestPredictionWorkflow(unittest.TestCase):
    """
    Test cases for the new unified prediction workflow using predict.py.
    """

    def setUp(self):
        """Set up the path to the prediction script under scripts/."""
        self.predictor_script = os.path.join('scripts', 'predict.py')
        if not os.path.exists(self.predictor_script):
            self.fail(f"Predictor script not found at: {self.predictor_script}")

    def test_script_runs_and_shows_help(self):
        """
        Test that the predict.py script runs and shows the help message.
        This is a basic check to ensure the script and its argument parser are working.
        """
        try:
                                                         
            result = subprocess.run(
                [sys.executable, self.predictor_script, '--help'],
                capture_output=True,
                text=True,
                check=True                                                                                  
            )
            
                                                        
            # Help message should display usage; avoid hard-coding script path text
            self.assertIn('usage:', result.stdout)
            self.assertIn('F1 Prediction System - Unified Workflow', result.stdout)
            self.assertIn('Available commands', result.stdout)
            self.assertIn('{fetch-data,train,predict}', result.stdout)
            
                                                           
            self.assertEqual(result.returncode, 0)
            
        except FileNotFoundError:
            self.fail(f"Failed to execute '{self.predictor_script}'. Ensure it is in the correct path and has execution permissions.")
        except subprocess.CalledProcessError as e:
            self.fail(f"Script execution failed with exit code {e.returncode}:\n{e.stderr}")

def run_tests():
    """Run the test suite."""
    print("Running F1 Prediction System Tests...")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(TestPredictionWorkflow))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Tests failed.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    if not run_tests():
        sys.exit(1) 