activate_this = 'C:/Users/SAMSUNG/Envs/food/Scripts/activate_this.py'
# execfile(activate_this, dict(__file__=activate_this))
exec(open(activate_this).read(),dict(__file__=activate_this))

import os
import sys
import site

# Add the site-packages of the chosen virtualenv to work with
site.addsitedir('C:/Users/SAMSUNG/myvenv/Lib/site-packages')




# Add the app's directory to the PYTHONPATH
sys.path.append('C:\Users\SAMSUNG\food')
sys.path.append('C:\Users\SAMSUNG\food\food')

os.environ['DJANGO_SETTINGS_MODULE'] = 'food.settings'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "food.settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()