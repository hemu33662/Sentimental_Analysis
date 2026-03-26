import os
import sys

# Set up Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "new_project.settings")

# Force DEBUG=True to see the traceback
os.environ["DJANGO_DEBUG"] = "true"

import django
django.setup()

from django.test import Client

c = Client()
try:
    response = c.post('/output', {'algo': 'Custom_LLM', 'text': 'Food is good'})
    print("Status Code:", response.status_code)
    print("Content:", response.content)
except Exception as e:
    import traceback
    traceback.print_exc()
