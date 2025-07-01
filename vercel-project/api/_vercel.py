from main import app
from mangum import Mangum

handler = Mangum(app, lifespan="off")  # Disable lifespan for Vercel