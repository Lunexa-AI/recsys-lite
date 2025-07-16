# Deployment Guide for Small Apps

This guide shows how to deploy vector-recsys-lite in production for small-scale apps (<10k users/items). It's designed to be simple and resource-efficient.

## Quick Setup

1. **Install**: `pip install vector_recsys_lite`
2. **Train Model**: Use CLI or API to create a model.pkl
3. **Deploy**: Use the `deploy` command or integrate manually

## One-Command Deployment

```bash
# Train a model (example)
vector-recsys predict ratings.csv --output model.pkl  # Actually save via API

# Generate API
vector-recsys deploy model.pkl --port 8000

# Run
uvicorn deploy_app:app --port 8000
```

Access http://localhost:8000/recommend/0 for recs.

## Integration Examples

### FastAPI (Advanced)

```python
from fastapi import FastAPI
from vector_recsys_lite import RecommenderSystem
import numpy as np

app = FastAPI()
rec = RecommenderSystem.load('model.pkl')

@app.get('/recommend/{user_id}')
def recommend(user_id: int, n: int = 5):
    # Get user ratings (from DB, etc.)
    user_ratings = np.zeros((1, num_items))  # Example
    preds = rec.predict(user_ratings)
    recs = rec.recommend(preds, n=n)[0]
    return {'recommendations': recs.tolist()}
```

### Script-Based Batch Processing

For offline recommendations:

```python
import pandas as pd
from vector_recsys_lite import RecommenderSystem

rec = RecommenderSystem().fit(ratings_df.values)
recs = rec.recommend(ratings_df.values)
pd.DataFrame(recs).to_csv('recommendations.csv')
```

## Scaling for Small Datasets
- **Memory Tips**: Use sparse matrices for >80% sparsity
- **Performance**: Enable Numba: `pip install numba`
- **Containerize**: Use our Dockerfile for cloud deployment
- **Monitoring**: Add simple logging to API

If datasets grow, consider migrating to larger libs like Implicit.

Questions? Open an issue.
