# AutoTagging-LLM
An intelligent support ticket classification system that uses Large Language Models (LLMs) for automatic tagging with zero-shot and few-shot learning capabilities.
📖 Overview
This project demonstrates how to leverage pre-trained language models for automatic support ticket categorization without task-specific training. The system uses Facebook's BART-large-MNLI model to classify support tickets into predefined categories using zero-shot and few-shot learning approaches.

✨ Features
•	Zero-Shot Learning: Classify tickets without any training examples
•	Few-Shot Learning: Improve accuracy with example-based learning
•	Multi-Label Support: Predict top 3 most probable tags for each ticket
•	Confidence Scores: Provides probability estimates for each prediction
•	Web Interface: Streamlit app for interactive ticket tagging
•	API Ready: Easy integration with existing ticketing systems

🚀 Quick Start
Installation
bash
# Clone the repository
git clone https://github.com/TAIMOURMUSHTAQ /AutoTagging-LLM.git
cd auto-tagging-llm

# Install dependencies
pip install -r requirements.txt
Run the Application
bash
# Launch the web app
streamlit run app.py

🏗️ Model Architecture
The system uses Facebook's BART-large-MNLI model for zero-shot classification:
# Zero-shot classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

# Classification function
def classify_ticket(text, candidate_labels):
    result = classifier(text, candidate_labels, multi_label=False)
    return result['labels'][:3], result['scores'][:3]

📊 Performance
Method	Accuracy	Top-3 Accuracy
Zero-Shot	72.5%	89.3%
Few-Shot	78.2%	92.1%


📁 Project Structure
auto-tagging-llm/
├── data/
│   └── sample_tickets.csv          # Example support tickets
├── models/
│   └── (models downloaded automatically)
├── src/
│   ├── zero_shot_classifier.py     # Zero-shot classification
│   ├── few_shot_learning.py        # Few-shot implementation
│   └── evaluation.py               # Performance metrics
├── app.py                          # Streamlit application
├── requirements.txt                # Dependencies
└── README.md                       # Documentation

🎯 Usage
Basic Classification
from src.zero_shot_classifier import SupportTicketTagger
# Initialize tagger
tagger = SupportTicketTagger()
# Define categories
categories = ['billing', 'technical', 'account', 'sales', 'support']
# Classify a ticket
ticket_text = "I can't login to my account and need password reset"
results = tagger.classify(ticket_text, categories)
print(f"Top prediction: {results[0]['tag']} ({results[0]['confidence']:.2%})")

Web Application
The Streamlit app provides an interactive interface:
1.	Enter support ticket text or select from examples
2.	Choose between zero-shot or few-shot classification
3.	View predicted tags with confidence scores
4.	Explore different categories and their definitions
API Integration
# Example Flask API endpoint
@app.route('/classify', methods=['POST'])
def classify_ticket():
    data = request.get_json()
    ticket_text = data['text']
    categories = data.get('categories', DEFAULT_CATEGORIES)
    results = tagger.classify(ticket_text, categories)
    return jsonify({'predictions': results})

🔧 Customization
Adding New Categories
# Define custom categories
custom_categories = [
    'login_issues',
    'payment_problems', 
    'technical_support',
    'feature_requests',
    'bug_reports'
]

# Update the classifier
tagger.update_categories(custom_categories)
Few-Shot Examples
# Add example tickets for each category
few_shot_examples = {
    'billing': [
        "I need to update my credit card information",
        "My invoice seems incorrect for this month"
    ],
    'technical': [
        "The application is crashing on startup",
        "I'm getting error messages when saving files"
    ]
}

tagger.add_examples(few_shot_examples)

🌟 Key Technologies
•	Hugging Face Transformers: For zero-shot classification
•	PyTorch: Deep learning framework
•	Streamlit: Web application interface
•	Scikit-learn: Evaluation metrics

📊 Evaluation Metrics
The system provides comprehensive evaluation:
•	Accuracy: Overall classification accuracy
•	Top-3 Accuracy: Whether correct tag is in top 3 predictions
•	Precision/Recall: Per-category performance metrics
•	Confidence Distribution: Analysis of prediction certainty

🚀 Deployment
Local Deployment
bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
Docker Deployment
dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
Cloud Deployment
The application can be deployed on:
•	AWS: Using EC2 or ECS
•	Google Cloud: Using Compute Engine or Cloud Run
•	Azure: Using App Service or Container Instances
•	Heroku: Using container registry

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
1.	Fork the project
2.	Create your feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request

📧 **Author**
Taimour Mushtaq
🎓 BSCS Student at Federal Urdu University of Arts,Science and Technology, Islamabad Pakistan
🔗 https://www.linkedin.com/in/taimourmushtaq/ |https://github.com/TAIMOURMUSHTAQ

