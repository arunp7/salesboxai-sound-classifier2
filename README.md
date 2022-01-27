This is the repository for SalesboxAI Sound Classifier for the following five sounds.

* Baby Cry
* Doorbell
* Rain
* Water Overflow
* Pressure Cooker

It is based on a simple convolutional neural network. The model is having an accuracy of 78.7%.

#### To build docker image
```
docker build -t sound-classifier .
```

### To run docker container 
```
docker run -d -p 5001:5001 sound-classifier
```
