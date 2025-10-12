from fastapi import FastAPI
api=FastAPI()
todo_dict = {
    1: {"task": "Morning exercise", "time": "7:00 AM", "done": False},
    2: {"task": "Breakfast", "time": "8:00 AM", "done": False},
    3: {"task": "Check emails", "time": "9:00 AM", "done": False},
    4: {"task": "Work on project", "time": "10:00 AM", "done": False},
    5: {"task": "Lunch break", "time": "1:00 PM", "done": False},
    6: {"task": "Team meeting", "time": "2:30 PM", "done": False},
    7: {"task": "Grocery shopping", "time": "5:00 PM", "done": False},
    8: {"task": "Dinner", "time": "7:00 PM", "done": False},
    9: {"task": "Read a book", "time": "8:30 PM", "done": False},
    10: {"task": "Plan next day", "time": "9:30 PM", "done": False}
}

todo=list(todo_dict.values())
print(todo)
# @api.get('/')
# def index():
#     return todo