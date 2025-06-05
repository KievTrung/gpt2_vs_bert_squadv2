from datasets import Dataset
from gpt2_vs_bert_rc_squadv2.inferencing import QaModel

if __name__ == "__main__":
    # sample dataset
    data = {
        "id": ["1", "2", "3"],
        "context": [
            "The sky is blue.",
            "Water boils at 100 degrees Celsius.",
            "Python is a programming language.",
        ],
        "question": [
            "What color is the sky?",
            "At what temperature does water boil?",
            "What is Python?",
        ],
    }
    samples = Dataset.from_dict(data)
    model = QaModel()
    result = model.pipeline(samples)

"""
    result has this format:
    result = { 'id':{
            "context": str,
            "question": str,
            "true answers":{'text': str, 'answer_start':int} 
            "pred": list 
        } 
    }
"""
