# Deep Learning for Computer Vision - Final Project

This project is part of the course Deep Learning for Computer Vision (20600) at Bocconi University. <br>
Creators: Botti, A.; Holzach, N.; Lorenzetti, S.; Mueller, J.; Puschiasis, P.; Schrock, F.

Please refer to `./face_analysis_presentation.pdf` and `./face_analysis_notebook.ipynb` for an overview of the project.

If you have any problems, questions, or suggestions, please contact me at jens.mueller@studbocconi.it


## Project description

Our final project combines several deep learning models (face detection, facial recognition, emotion recognition)
to create a marketing application that can recognize reccuring customers in a store and analyze the effect
of marketing activities by recognizing the customers' emotions at different points in the store.


## Repository structure

The project repository contains four folders. The first three comprise the individual project parts face detection,
facial recognition, and emotion recognition. The fourth folder combines all three individual parts into a single
fully-integrated program.

The repository structure is shown below.
Full-size models and datasets are available on request.

```
face_analysis
│   README.txt
│   face_analysis_presentation.pdf
│   face_analysis_notebook.ipynb
│   requirements.txt
│   requirements_minimal.txt
│   .gitignore
│
├───emotion_recognition
│   ├───benchmarking
│   ├───emotion_model_csv
│   ├───example_images
│   └───models
├───face_detection
│   ├───checkpoints
│   ├───configs
│   └───data
├───facial_recognition
│   ├───data
│   ├───models
│   ├───openface
│   └───trained_model
└───full_project_integration
    └───data
```

## Requirements

For the repository to run correctly and completely, all libraries from the requirements.txt file must be installed.

**For Windows users**:
Before installing dlib (and the corresponding face_recognition library),
Cmake and Visual Studio must must downloaded to be able to compile from C++.
Alternatively, you can install `requirements_minimal.txt` (without dlib) for partial functionality.
