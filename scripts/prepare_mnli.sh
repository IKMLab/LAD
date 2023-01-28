# Download MNLI dataset.
wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip

# Move MNLI into data folder.
mv MNLI.zip ./data/fine_tune/MNLI.zip

# Extract MNLI from zip.
unzip ./data/fine_tune/MNLI.zip -d ./data/fine_tune/

# Remove redundant file.
rm ./data/fine_tune/MNLI.zip