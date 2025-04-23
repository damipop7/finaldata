import os

# Define the folder structure
folders = [
    "data_pipeline",
    "clients",
    "tests",
    "configs"
]

files = {
    "data_pipeline": ["__init__.py", "ingestion.py", "preprocessing.py", "analysis.py", "modeling.py", "evaluation.py"],
    "clients": ["kaggle_client.py", "rapidapi_client.py"],
    "tests": ["test_ingestion.py", "test_preprocessing.py", "test_analysis.py", "test_modeling.py", "test_evaluation.py"],
    "configs": ["config.yaml"],
    "": ["main.py", "README.md", "requirements.txt", ".gitignore"]
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for folder, file_list in files.items():
    for file in file_list:
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                # Add minimal placeholder code
                if file.endswith(".py"):
                    f.write("# " + file.replace(".py", "").capitalize() + "\n")
                elif file == "README.md":
                    f.write("# Big Data Analytics in Soccer\n")
                elif file == "requirements.txt":
                    f.write("pandas\nnumpy\nrequests\npyyaml\nscikit-learn\npytest\nkaggle\n")
                elif file == ".gitignore":
                    f.write("venv/\n__pycache__/\n*.pyc\n.env\n")
                elif file == "config.yaml":
                    f.write("kaggle_api_key: 'your_kaggle_key_here'\nrapidapi_key: 'your_rapidapi_key_here'\n")

print("✅ Project structure created successfully!")
