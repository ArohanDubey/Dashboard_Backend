from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import openai
import os
import re
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv('OPENAI_API_KEY')

app.config['static'] = 'static'  # Directory for HTML files
os.makedirs(app.config['static'], exist_ok=True)  # Ensure the HTML folder exists

# Global variable to store the dataframe
df = None

MAX_RETRIES = 4
retries = 0

@app.route('/')
def home():
    return "Welcome to the Data Analytics API! Use the /upload and /analyze endpoints."

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    global MAX_RETRIES
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        return jsonify({'message': 'File uploaded successfully'})
    else:
        return jsonify({'error': 'Invalid file type, please upload a CSV file'}), 400

@app.route('/analyze', methods=['GET'])
def analyze_data():
    global df
    if df is None:
        return jsonify({'error': 'No data uploaded. Please upload a CSV file first.'}), 400
    try:
        plotly_code = get_plotly_code_from_gpt(df)
        html_file_path = execute_plotly_code(plotly_code, df, 0)
        return send_from_directory(app.config['static'], "enhanced_interactive_graph.html")
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_plotly_code_from_gpt(dataframe, error_message=None):
    data_description = request.args.get('input')
    app.logger.info(data_description)
    prompt = f"""
    Given the following data description:
    {data_description}
    
    Generate Python code using Plotly to create an enhanced interactive. 
    The graph should be colorful, attractive, and appealing. 
    It should include dynamic range colors, annotations for key points, and interactivity.
    The data contains columns representing different metrics. Use appropriate graph types based on data characteristics. 
    Make Sure that that the generated code is not a example but actual code and also re-verify your code before generation.
    
    **Instructions**
    - Use this Dataframe in your generated code instead of Giving any sample value: {dataframe}
    """
    
    if error_message:
        prompt += f"\nPrevious attempt resulted in the following error: {error_message}"

    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            code_block = response['choices'][0]['message']['content'].strip()
            print("Generated Code Block:\n", code_block)
            
            code_match = re.search(r"```python(.*?)```", code_block, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                
                # Ensure necessary imports
                if 'make_subplots' not in code:
                    code = "from plotly.subplots import make_subplots\n" + code
                if 'numpy' not in code:
                    code = "import numpy as np\n" + code
                if 'plotly.graph_objects as go' not in code:
                    code = "import plotly.graph_objects as go\n" + code
                
                return code
            else:
                raise ValueError("Code block not found in the response.")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise

def execute_plotly_code(coder, dataframe, retires = 0):
  

    if  retires < MAX_RETRIES:
        try:
            code = coder.replace("fig.show()", "")
            local_vars = {'df': dataframe, 'px': px, 'np': np, 'make_subplots': make_subplots, 'go': go}
            exec(code, {}, local_vars)
            fig = local_vars['fig']

            static_dir = os.path.join(app.root_path, 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)

            html_file_path = os.path.join(static_dir, 'enhanced_interactive_graph.html')
            app.logger.info(f"Saving HTML file to: {html_file_path}")
            fig.write_html(html_file_path)

            return html_file_path
        except Exception as e:
            retires += 1
            error_message = str(e)
            print(f"Retry {retires} failed with error: {error_message}")
            coder = get_plotly_code_from_gpt(df, error_message=error_message)
            return execute_plotly_code(coder,df, retires)
    else :
        raise Exception(f"Failed after {MAX_RETRIES} retries: {error_message}")

if __name__ == '__main__':
    app.run(debug=True)
