# Kinship-relationship

This project contains the source code for finding facial similarity between to individuals sharing blood relation and not.
Inorder to train the model, used the dataset with a root directory, individual directory inside root directory for each family. Within each family directory, generate a directory for each individual in the family and place the images inside it.
Also use a csv file that represents the image pairs that shares a similarity. Unrelated pairs will be generated in random.
The VGG architecture of CNN is used as a Siamese network with identical CNN models to perform the feature extraction.
After training, keep a copy of the trained model inside kinship directory for using the GUI. This GUI is designed using Django.
For running the GUI, execute 'python manage.py runserver' in the command prompt, uload the images in the resulting web page. The output class along with confidence score will be displayed.
