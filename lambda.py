import json
import boto3
import base64

### Serialize Image Data

s3 = boto3.client('s3')

# For each function, we will have to change the Handler to match the names below
def lambda_serialize_image_data_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        "statusCode": 200,
        "body": {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


### Classify Images

# Fill this in with the name of your deployed model
model_endpoint = 'scones-bicycle-motorcycle-classifier-2022-01-18'

# Use the Sagemaker runtime API to call our model endpoint
# This was suggested by a mentor on https://knowledge.udacity.com/questions/748461
sagemaker_client = boto3.client('runtime.sagemaker')

def lambda_classify_image_handler(event, context):
    """A function to classify image data"""
    
    # Decode the image data
    image = base64.b64decode(event["image_data"])
    
    response = sagemaker_client.invoke_endpoint(
        EndpointName = model_endpoint,
        ContentType = 'image/png',
        Body = image
    )
    
    # We return the data back to the Step Function    
    event["inferences"] = response["Body"].read().decode('utf-8')
    return {
        'statusCode': 200,
        'body': event
    }


### Filter Low-Confidence Inferences

threshold = .75

def lambda_filter_inference_handler(event, context):
    """A function to filter low-confidence inferences"""
    
    # Grab the inferences from the event
    inferences = event["inferences"][1:-1].split(', ')
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = False
    if (float(inferences[0]) >= threshold):
        meets_threshold = True
    elif (float(inferences[1]) >= threshold):
        meets_threshold = True

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise LowConfidenceException

    return {
        'statusCode': 200,
        'body': event
    }
    
class LowConfidenceException(Exception):
    """Exception raised for inferences with too low confidence."""

    def __init__(self):
        super().__init__("Threshold confidence not met.")