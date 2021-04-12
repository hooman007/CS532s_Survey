#preparing for download
#source: https://gist.github.com/KarthikMAM/d8ebde4db84a72b083df0e14242edb1a
for i in {1..19}
do
  printf "\n\n------------------------- Downloading $i th speaker -------------------------\n\n"
  #download the audio of the ith speaker
  curl "https://zenodo.org/record/3625687/files/s$i.zip?download=1" > "s$i.zip" && unzip "s$i.zip"
  cd "s$i" && mkdir "inputs" && mkdir "labels" && cd ..
done
