file(GLOB headers *.h) # creates a set headers with all the .h files
file(GLOB sources *.cpp) # creates a set sources with all the .cpp files

# Create a set with all the files in this folder and send it to parent folder with the command PARENT_SCOPE
set(nnlib_files ${headers} ${sources} PARENT_SCOPE) 