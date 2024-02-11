import numpy as np

class Portfolio:

    @classmethod
    def compute_portfolio(cls, weights, prices, initial_capital=1000000):
        '''
        Compute rebalanced allocations for each assset on each period based on weights and prices. 
        Rebalancing is calculated based on portfolio value (price x previous allocations) then compute a new allocation
        based on given weights.

        Returns - allocations (num_shares), portfolio_value for each period.
        '''
        assert weights.shape == prices.shape, 'Shape of weights and prices must be the same'
        allocations = np.zeros(shape=weights.shape) # additional column for cash
        current_value = initial_capital

        for i in range(weights.shape[0]):
            current_value = current_value or (allocations[i-1,:] * prices[i,:]).sum(axis=0)
            value_alloc = current_value * weights[i,:]
            share_alloc = value_alloc / prices[i,:]
            allocations[i] = share_alloc
            print('CURRENT', current_value)
            print( ' current_value', (allocations[i-1,:] * prices[i,:]))
            print('  value_alloc', value_alloc)
            print('  portfolio', allocations[i])
            print('  price', prices[i])
            current_value = None
        
        return allocations, (allocations * prices).sum(axis=0)
