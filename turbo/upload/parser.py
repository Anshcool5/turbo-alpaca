import json
from typing import Any, Dict

def parse_business_idea(json_str: str) -> Dict[str, Any]:
    """
    Parse a JSON string containing business idea details into a dictionary.

    Args:
        json_str (str): A JSON-formatted string representing the business idea.

    Returns:
        dict: A dictionary with keys like "name", "explanation", and "how_to_set_up".
    """
    try:
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return {}

# Example usage
if __name__ == "__main__":
    json_input = """
{
    "name": "EduCode",
    "explanation": "EduCode is an online platform that provides interactive coding lessons and project-based learning experiences for students of all ages. The platform will utilize a combination of video tutorials, interactive coding exercises, and real-world projects to teach programming concepts. The business will align with the candidate's skills in Java, Python, Django, and data analysis, as well as their experience in teaching and mentoring. The platform will also incorporate features such as role-based access control, secure payment gateways, and data analytics to track student progress and engagement.",
    "how_to_set_up": [
        {
            "step": 1,
            "title": "Define Target Market",
            "description": "Identify the target audience for EduCode, including age range, skill level, and programming interests. Conduct market research to understand the demand for online coding education and the competitive landscape."
        },
        {
            "step": 2,
            "title": "Develop Curriculum",
            "description": "Create a comprehensive curriculum for EduCode, covering topics such as introductory programming, data structures, and web development. Develop interactive coding exercises and real-world projects to reinforce learning concepts."
        },
        {
            "step": 3,
            "title": "Design and Develop Platform",
            "description": "Design and develop the EduCode platform using Django, incorporating features such as role-based access control, secure payment gateways, and data analytics. Utilize modern JavaScript frameworks to create a responsive and intuitive user interface."
        },
        {
            "step": 4,
            "title": "Establish Mentorship Program",
            "description": "Establish a mentorship program that connects students with experienced programmers and industry professionals. Recruit mentors and develop a system for matching students with mentors based on their interests and skill levels."
        },
        {
            "step": 5,
            "title": "Launch Marketing Campaign",
            "description": "Launch a marketing campaign to promote EduCode and attract students. Utilize social media, online advertising, and content marketing to reach the target audience and build a community around the platform."
        },
        {
            "step": 6,
            "title": "Monitor and Evaluate Progress",
            "description": "Monitor and evaluate student progress and engagement on the EduCode platform. Use data analytics to track key metrics such as student retention, completion rates, and satisfaction. Make data-driven decisions to improve the platform and curriculum over time."
        }
    ]
}
"""
    
    parsed_idea = parse_business_idea(json_input)
    print("Business Idea Name:", parsed_idea.get("name"))
    print("Explanation:", parsed_idea.get("explanation"))
    print("Setup Steps:")
    for step in parsed_idea.get("how_to_set_up", []):
        print(f"  Step {step.get('step')}: {step.get('title')} - {step.get('description')}")
