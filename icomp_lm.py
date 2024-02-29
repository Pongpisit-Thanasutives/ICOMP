import numpy as np
import statsmodels.api as sm
# Implemented by Pongpisit Thanasutives in 2024

def llf_complexity(X_pre, y_pre, coef=None, a_n=None, include_bias=False):
    N = len(y_pre)
    if a_n is None:
        a_n = np.log(N)

    if include_bias:
        X_pre = sm.add_constant(X_pre)
    if coef is None:
        model = sm.OLS(y_pre, X_pre)
        m_res = model.fit()
        q = model.rank
        rss = np.sum(m_res.resid**2)
        S_inv = m_res.cov_params(scale=1)
        llf = m_res.llf
    else:
        q = np.linalg.matrix_rank(X_pre)
        rss = np.sum((y_pre-X_pre@beta)**2)
        S_inv = np.linalg.inv(X_pre.T@X_pre)
        llf = -0.5*(N*np.log(2*np.pi) + N*np.log(rss/N) + N)

    C0 = np.trace(np.log(S_inv))-np.log(np.linalg.det(S_inv))
    C1 = q*np.log(np.trace(S_inv)/q)-np.log(np.linalg.det(S_inv))
    C_IFIM = (q+1)*np.log((np.trace(S_inv) + 2*rss/N)/(q+1)) - \
                np.log(np.linalg.det(S_inv)) - np.log(2*rss/N)
    C_COV = (q+1)*np.log((np.trace(S_inv) + 2*rss*(N-q)/(N**2))/(q+1)) - \
                np.log(np.linalg.det(S_inv)) - np.log(2*rss*(N-q)/(N**2))

    C0 = C0/2
    C1 = C1/2
    C_IFIM = C_IFIM/2
    C_COV = C_COV/2
    
    complexities = np.array([C0, C1, C_IFIM, C_COV])
    icomps = -2*llf + 2*a_n*complexities

    return llf, complexities, icomps

