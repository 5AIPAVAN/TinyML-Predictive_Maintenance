# from flask import Flask, render_template, request
# import numpy as np
# import tensorflow as tf

# app = Flask(__name__)

# # Load the TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
# import numpy as np

# def preprocess_input(air_temp, process_temp, speed, torque, tool_wear):
#     # Convert string inputs to appropriate data types
#     try:
#         air_temp = float(air_temp)
#         process_temp = float(process_temp)
#         speed = float(speed)
#         torque = float(torque)
#         tool_wear = float(tool_wear)

#         # Construct input tensor in the shape expected by the model
#         input_data = np.array([[air_temp, process_temp, speed, torque, tool_wear]], dtype=np.float32)
#         return input_data
#     except ValueError:
#         # Handle the case where conversion fails (e.g., empty string)
#         return None

# # Define function for post-processing output
# def postprocess_output(prediction):
#     # Perform any necessary postprocessing
#     return prediction

# @app.route("/", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         form_inputs = request.form.to_dict()
#         input_data = preprocess_input(**form_inputs)

#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()

#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])

#         prediction = postprocess_output(output_data)

#         return render_template("result.html", prediction=prediction)
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess input data
def preprocess_input(air_temperature, process_temperature, rotational_speed, torque, tool_wear):
    # Convert string inputs to appropriate data types
    try:
        air_temperature = float(air_temperature)
        process_temperature = float(process_temperature)
        rotational_speed = float(rotational_speed)
        torque = float(torque)
        tool_wear = float(tool_wear)

        # Construct input tensor in the shape expected by the model
        input_data = np.array([[air_temperature, process_temperature, rotational_speed, torque, tool_wear]], dtype=np.float32)
        return input_data
    except ValueError:
        # Handle the case where conversion fails (e.g., empty string)
        return None

# Function to make predictions
def predict_failure(input_data):
    # Preprocess input data
    input_data = preprocess_input(*input_data)
    
    if input_data is None:
        raise ValueError("Invalid input data")
    
    # Ensure input data matches the expected shape of the TFLite model
    input_shape = input_details[0]['shape']
    
    if input_data.shape != input_shape:
        raise ValueError(f"Input shape mismatch. Expected: {input_shape}, Got: {input_data.shape}")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output[0][0]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    air_temperature = request.form['air_temperature']
    process_temperature = request.form['process_temperature']
    rotational_speed = request.form['rotational_speed']
    torque = request.form['torque']
    tool_wear = request.form['tool_wear']
    
    # Make prediction
    prediction = predict_failure([air_temperature, process_temperature, rotational_speed, torque, tool_wear])
    
    # Determine the result
    if prediction >= 0.5:
        result = "Failure"
    else:
        result = "No Failure"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
