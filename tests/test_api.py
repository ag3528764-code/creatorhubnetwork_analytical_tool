from app.app import app
def test_health():
 app.config['TESTING']=True
 with app.test_client() as c: assert c.get('/api/health').status_code==200
