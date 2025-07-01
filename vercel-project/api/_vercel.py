from main import app
from mangum import Mangum

# Explicitly disable lifespan for Vercel compatibility
handler = Mangum(app, lifespan="off")