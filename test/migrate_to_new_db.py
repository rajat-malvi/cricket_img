"""
Master script to migrate data to the new PostgreSQL database for the Cricket Image Chatbot
"""

import os
import sys
import time

def print_step(step_number, step_description):
    """
    Print a step header
    """
    print("\n" + "=" * 80)
    print(f"STEP {step_number}: {step_description}")
    print("=" * 80)

def run_script(script_name):
    """
    Run a Python script and check the exit code
    """
    print(f"Running {script_name}...")
    exit_code = os.system(f"python {script_name}")
    
    if exit_code != 0:
        print(f"Error running {script_name}. Exit code: {exit_code}")
        return False
    
    return True

def main():
    """
    Main function to run all migration steps
    """
    print("Starting migration to new database 'jsk1_data'...")
    
    # Step 1: Create the new database
    print_step(1, "Creating new database")
    if not run_script("create_new_db.py"):
        print("Failed to create new database. Aborting migration.")
        return
    
    # Step 2: Migrate data to the new database
    print_step(2, "Migrating data to new database")
    if not run_script("migrate_data.py"):
        print("Failed to migrate data. Aborting migration.")
        return
    
    # Step 3: Verify the database setup
    print_step(3, "Verifying database setup")
    if not run_script("verify_db.py"):
        print("Database verification failed. Migration may be incomplete.")
        return
    
    # Step 4: Run the application
    print_step(4, "Running the application")
    print("Migration completed successfully!")
    print("\nYou can now run the application with:")
    print("python run.py")

if __name__ == "__main__":
    main()
