import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(20)

def generate_distributions(sample_size=1000):
    """
    Generate sample data and their probability density functions for normal, 
    uniform, and exponential distributions with default parameters.
    """
    distributions = {
        'Normal': {
            'data': np.random.normal(0, 1, size=sample_size),
            'pdf': lambda x: stats.norm.pdf(x, 0, 1)
        },
        'Uniform': {
            'data': np.random.uniform(0, 1, size=sample_size),
            'pdf': lambda x: stats.uniform.pdf(x, 0, 1)
        },
        'Exponential': {
            'data': np.random.exponential(1, size=sample_size),
            'pdf': lambda x: stats.expon.pdf(x, 0, 1)
        }
    }

    return distributions

distributions = generate_distributions()
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle('Boxplots and normalized histograms with probability density functions', fontsize=14, fontweight='bold')

for idx, (name, dist) in enumerate(distributions.items()):
    data = dist['data']
    x = np.linspace(data.min(), data.max(), 100)
    
    # Boxplot
    ax[0, idx].boxplot(data, orientation='horizontal')
    ax[0, idx].set_title(f'{name} data boxplot')
    
    # Histogram with distribution curve
    ax[1, idx].hist(data, bins=40, density=True, alpha=0.7, color='lightskyblue', edgecolor='black')
    ax[1, idx].plot(x, dist['pdf'](x), 'r-', linewidth=2)
    ax[1, idx].set_title(f'{name} data histogram and PDF')
    ax[1, idx].set_ylabel('Density')

plt.tight_layout()
plt.savefig('boxplots-for-popular-distributions.png', dpi=300)
print("Figure saved as boxplots-for-popular-distributions.png")