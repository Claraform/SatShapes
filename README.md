////////////////////////////////////////////////////////////////////////
//                                                                    //
//      SATSHAPES - MAPPING SATELLITE GIS IMAGE TO VECTOR SHAPES      //
//                                                                    //
////////////////////////////////////////////////////////////////////////

Clara Esther Stassen
STSCLA001

An application developed for EEE4022S to detect shapes in GIS images and 
provide a vector image output of the shapes found in the image along with
accompanying data of the geographic coordinates and dimensions of the 
detected shapes. 

I. File Structure
------------------
conversion.py     Main driver file: pre-processes, processes, and post-processes input data
shapes.py         Contains functionality for shape detection

III. How To Run
----------------
- To run conversion.py:
    $ python3 conversion.py -i <input_image> -c <coordinate_file>

IV. Arguments
--------------
-i    Path to input bitmap GIS image
-c    Path to input coordinate text file

V. Inputs
--------------
1. A bitmap GIS image (.png, .jpg)
2. A textfile containing the coordinates of the top left and bottom right
    corners of the input image (.txt). The format is:
              latitude1,longitude1,latitude2,longitude2

VI. Outputs
--------------
1. output.png         A copy of the input image with detected shapes outlined
2. outline.png        A binary template of the detected shapes
3. output.txt         List of detected shapes and their coordinates and dimensions
4. vector_output.eps  Vector image of detected shapes

VI. Notes
--------------
- Image filtering techniques can be tuned by altering the following variables
    in the main() method in conversion.py:
          - blurring # Sets image blurring method
          - edge_detect # Sets edge detection method
          - sharpen # Sets sharpening method
- Contrast can be adjusted by tuning the parameters alpha and beta in preprocess()
   in conversion.py
- The shape detection method can be set by changing the shape variable in main()
   in conversion.py