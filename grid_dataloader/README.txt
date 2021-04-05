To get the dataloader working:

1) Load the grid data into the directories that are layed out here: from s1 through s9 (feel free to add more)

2) Run the input_processing script for each of the directories, s1 -> s9, tweaking the parameters in the file to optimize the results for the specific speaker

3) Run the output_processing script for each of the directories, s1 -> s9

4) Incorporate the dataloader and skeleton training loop laid out in grid_dataloader.py into your own training loop