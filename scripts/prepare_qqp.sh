# Download QQP dataset.
wget https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip

# Move QQP into data folder.
mv QQP-clean.zip ./data/fine_tune/QQP-clean.zip

# Extract QQP from zip.
unzip ./data/fine_tune/QQP-clean.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/QQP-clean.zip