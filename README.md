# Mixed models

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

> Reference implementations for (generalized) linear mixed models.

## About

Concise reference implementations for fitting frequentist (generalized) linear mixed models. 
The implementations are not really efficient and don't use results from contemporary research (i.e., `lme4`'s PLS and PIRLS).
Specifically, the following methods are used:

- For LMMs first the random effects are marginalized out and the variance parameters of the errors and of the random effects are estimated.
Having the variance components estimated, Henderson's mixed model equations are used to estimate the fixed effects and predict random effects.  

- For GLMMs an IRLS approach is used. In the first step of each iteration variance components are estimated by using a Laplace approximation to the likelihood of the marginal model.
In the second step we estimate the fixed effects and predict random effects using Henderson's mixed mdoel equations on working responses.
 
## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier @ web.de</a>
