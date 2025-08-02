#!/usr/bin/env python3
"""
Test runner for the email functionality
Run this script to execute all tests for the email system
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests with detailed output"""
    print("🧪 Running Email System Tests")
    print("=" * 50)
    
    # Change to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "server/test_email.py", 
            "-v",
            "--tb=short",
            "--color=yes"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code {result.returncode}")
            
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_specific_test(test_name):
    """Run a specific test"""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            f"server/test_email.py::{test_name}", 
            "-v",
            "--tb=short",
            "--color=yes"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'pytest': 'pytest',
        'pytest-mock': 'pytest_mock', 
        'responses': 'responses',
        'requests': 'requests',
        'google-generativeai': 'google.generativeai'
    }
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} (missing)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run email system tests")
    parser.add_argument("--test", help="Run a specific test")
    parser.add_argument("--check-deps", action="store_true", help="Only check dependencies")
    
    args = parser.parse_args()
    
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)