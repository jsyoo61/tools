# tools-jsyoo61

(`tools` package was replaced on 2025.04.28. If you're looking for web scraping tools, visit http://pypi.python.org/pypi/weblib)

Python syntax tools for faster programming. [github](https://github.com/jsyoo61/tools) \
jsyoo61@unc.edu

    pip install tools

    import tools as T
    T.save_pickle(data, 'data.p')
    data = T.load_pickle('data.p')

Most useful functions are in `tools.tools` which is loaded to the main module by default. \
API dependent tools are in their corresponding folders: `tools.API_name` (i.e. `tools.torch.optim`)

Undocumented, so you'll have to look at documents written inside the codes or with help(). (Most codes contain explanation)
Planning to document with chatGPT.

I'm happy to discuss questions or suggestions, please contact me via email. :)

1.0.1. (2025.5.1.)
- `tools.sklearn.metrics.r2_score()` `multioutput` argument fixed

1.0.2. (2025.5.3.)
- `tools.sklearn.metrics` axis=None returns a single np.float object