# Download RTE dataset.
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip

# Move RTE into data folder.
mv RTE.zip ./data/fine_tune/RTE.zip

# Extract RTE from zip
unzip ./data/fine_tune/RTE.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/RTE.zip