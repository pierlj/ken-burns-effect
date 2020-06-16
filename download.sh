#!/bin/bash
dldir="./models/trained"

### Check for dir, if not found create it using the mkdir ##
[ ! -d "$dldir" ] && mkdir -p "$dldir"
 
# Now download it
wget -qc "$url" -O "${dldir}/${file}"
echo 'Starting download of the trained networks...'

fileid='1pnkRqUKS2T4QZ161RCToykeK0Z3BL271' 
filename='disparity-refinement.tar'
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${fileid}" -O "${dldir}/${filename}"

fileid='1OV46pzDl29fft13ZNJEvkAC6AjU3fvyc' 
filename='inpainting-color.tar'
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${fileid}" -O "${dldir}/${filename}"

fileid='1s8lPOMVK4eTb5gA5_huBmbUMAeOBG-CA' 
filename='inpainting-depth.tar'
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${fileid}" -O "${dldir}/${filename}"

fileid='1185R-YeKRyBulMQmW-91sfx5-y8ShloW' 
filename='inpainting-pretrained.tar'
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${fileid}" -O "${dldir}/${filename}"

fileid=1qlzhkfC58zsnEPZdQDukHEBIEKJB-6Mc 
filename='disparity-estimation-no-mask.tar'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qlzhkfC58zsnEPZdQDukHEBIEKJB-6Mc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qlzhkfC58zsnEPZdQDukHEBIEKJB-6Mc" -O "${dldir}/${filename}" && rm -rf /tmp/cookies.txt

fileid=13Y6-hdM8MEDBRmv0owwjtYScBiu5aOw3 
filename='disparity-estimation-mask.tar'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13Y6-hdM8MEDBRmv0owwjtYScBiu5aOw3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13Y6-hdM8MEDBRmv0owwjtYScBiu5aOw3" -O "${dldir}/${filename}" && rm -rf /tmp/cookies.txt

echo 'Download completed.'