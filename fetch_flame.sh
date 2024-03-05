#!/bin/bash

# Fetch FLAME data
echo -e "\nBefore you continue, you must register at https://flame.is.tue.mpg.de/ and agree to the FLAME license terms."

username='majorabdo12@gmail.com'
password='wuhpu5-pYqqid-bymbuz'


echo -e "\nDownloading FLAME..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1' -O './models/FLAME2023.zip' --no-check-certificate --continue
unzip ./FLAME2023.zip -d ./flame_model/assets/flame/
rm -rf FLAME2023.zip

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './models/FLAME2020.zip' --no-check-certificate --continue
unzip ./FLAME2020.zip -d ./flame_model/assets/flame/
rm -rf FLAME2020.zip


wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' -O './FLAME_masks.zip' --no-check-certificate --continue
unzip FLAME_masks.zip -d ./flame_model/assets/flame/

rm -rf FLAME_masks.zip
