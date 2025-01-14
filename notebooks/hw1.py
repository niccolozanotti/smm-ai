import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""# Homework 1 : Linear Algebra and Floating Point Arithmetic""")
    return


@app.cell
def _(check_dependencies):
    check_dependencies()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Direct Methods for the solution of Linear Systems

        1. Given a matrix $A \in \mathbb{R}^{n \times n}$, the vector $x_{true} = (1,1,...,1)^T \in \mathbb{R}^n$, and a value for $n$, write a script that:
            - Computes the right-hand side of the linear system $\bm{y} = A \bm{x_{true}}$ (test problem).
            - Computes the condition number in $2$-norm of the matrix $A$. It is ill-conditioned? What if we use the $\infty$-norm instead of the 2-norm?
            - Solves the linear system $A \bm{x} = \bm{y}$ with the function `np.linalg.solve()`.
            - Computes the relative error between the computed solution and the true solution $x_{true}$.
            - Plot a graph (using `matplotlib.pyplot`) with the relative errors as a function of $n$ and (in a different window) the condition number in $2$-norm and in $\infty$-norm, as a function of $n$.
        """
    )
    return


@app.cell
def _(cond, norm, np, pl, solve):
    """Functions for analyzing linear systems."""

    def test_matrix(A: np.ndarray) -> dict:
        """Analyze a linear system Ax = b with known solution.

        Parameters
        ----------
        A : np.ndarray
            Input matrix of shape (n, n)

        Returns
        -------
        dict
            Analysis results including condition numbers and errors
        """
        n = A.shape[0]
        x_true = np.ones((n, 1))
        b = A @ x_true

        cond_2norm = cond(A, 2)
        cond_infnorm = cond(A, np.inf)

        x_computed = solve(A, b)
        rel_err = norm(x_computed - x_true) / norm(x_true)

        return {
            'size': n,
            'cond_2norm': cond_2norm,
            'cond_infnorm': cond_infnorm,
            'relative_error': rel_err,
        }

    def analyze_matrix_family(n_values: list, matrix_generator, name: str) -> pl.DataFrame:
        """Analyze a family of matrices for different sizes.

        Parameters
        ----------
        n_values : list
            List of matrix sizes to test
        matrix_generator : callable
            Function that generates matrix of given size
        name : str
            Name of the matrix family

        Returns
        -------
        pl.DataFrame
            DataFrame containing analysis results
        """
        results = []
        for n in n_values:
            result = test_matrix(matrix_generator(n))
            results.append({
                'size': n,
                'cond_2norm': result['cond_2norm'],
                'cond_infnorm': result['cond_infnorm'],
                'relative_error': result['relative_error'],
                'matrix_type': name
            })

        return pl.DataFrame(results)
    return analyze_matrix_family, test_matrix


@app.cell
def _(hilbert, np):
    """Matrix generators for different test cases."""

    def random_matrix(n: int) -> np.ndarray:
        """Generate random matrix using np.random.rand."""
        return np.random.rand(n, n)

    def vander_matrix(n: int) -> np.ndarray:
        """Generate Vandermonde matrix for vector [1,2,...,n]."""
        v = np.arange(1, n + 1)
        return np.vander(v)

    def hilbert_matrix(n: int) -> np.ndarray:
        """Generate Hilbert matrix of size n."""
        return hilbert(n)
    return hilbert_matrix, random_matrix, vander_matrix


@app.cell
def _(
    analyze_matrix_family,
    hilbert_matrix,
    pl,
    random_matrix,
    vander_matrix,
):
    """Generate and visualize results."""

    # Generate results for each matrix type
    n_random = list(range(10, 110, 10))
    n_vander = list(range(5, 35, 5))
    n_hilbert = list(range(4, 13))

    results_random = analyze_matrix_family(n_random, random_matrix, "Random")
    results_vander = analyze_matrix_family(n_vander, vander_matrix, "Vandermonde")
    results_hilbert = analyze_matrix_family(n_hilbert, hilbert_matrix, "Hilbert")

    # Combine all results
    all_results = pl.concat([results_random, results_vander, results_hilbert])
    return (
        all_results,
        n_hilbert,
        n_random,
        n_vander,
        results_hilbert,
        results_random,
        results_vander,
    )


@app.cell
def _(all_results, pl, plt):
    def generate_plots(results: pl.DataFrame) -> tuple[plt.Figure, plt.Figure]:
        """
        Create publication-quality matplotlib figures.

        Parameters
        ----------
        results : pl.DataFrame
            DataFrame containing analysis results

        Returns
        -------
        tuple[plt.Figure, plt.Figure]
            Figures for condition numbers and relative errors
        """
        # Create figure for condition numbers
        fig_cond = plt.figure(figsize=(8, 6))
        for matrix_type in results['matrix_type'].unique():
            data = results.filter(pl.col('matrix_type') == matrix_type)
            plt.semilogy(data['size'], data['cond_2norm'], 
                        'o-', label=matrix_type, markersize=6)

        plt.xlabel('Matrix Size $(n)$')
        plt.ylabel(r'Condition Number $\kappa_2(A)$')
        plt.title(r'Condition Numbers vs Matrix Size')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        # Create figure for relative errors
        fig_err = plt.figure(figsize=(8, 6))
        for matrix_type in results['matrix_type'].unique():
            data = results.filter(pl.col('matrix_type') == matrix_type)
            plt.semilogy(data['size'], data['relative_error'], 
                        'o-', label=matrix_type, markersize=6)

        plt.xlabel('Matrix Size $(n)$')
        plt.ylabel(r"Relative Error $|| \bm{x} - \bm{x}_{true}||_2 / |\bm{x}_{true} |_2$")
        plt.title('Relative Errors vs Matrix Size')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        return fig_cond, fig_err

    figs = generate_plots(all_results)

    # Save the figures
    figs[0].savefig('condition_numbers.pdf', bbox_inches='tight', dpi=300)
    figs[1].savefig('relative_errors.pdf', bbox_inches='tight', dpi=300)
    figs[0]
    return figs, generate_plots


@app.cell
def _(figs):
    figs[1]
    return


@app.cell
def _(all_results, pl):
    """Print analysis summary."""

    def print_detailed_analysis(results: pl.DataFrame) -> str:
        """Generate detailed analysis summary.

        Parameters
        ----------
        results : pl.DataFrame
            DataFrame containing analysis results

        Returns
        -------
        str
            Formatted analysis summary
        """
        analysis = []
        threshold = 1e6

        for matrix_type in results['matrix_type'].unique():
            type_results = results.filter(pl.col('matrix_type') == matrix_type)

            max_cond = type_results['cond_2norm'].max()
            min_cond = type_results['cond_2norm'].min()
            max_error = type_results['relative_error'].max()

            analysis.append(f"\n=== {matrix_type} Matrix Analysis ===")
            analysis.append(f"Condition Number Range: {min_cond:.2e} to {max_cond:.2e}")
            analysis.append(f"Maximum Relative Error: {max_error:.2e}")

            if max_cond > threshold:
                analysis.append(f"WARNING: Matrix becomes ill-conditioned (κ > {threshold:.1e})")
                ill_cond_size = type_results.filter(pl.col('cond_2norm') > threshold)
                if not ill_cond_size.is_empty():
                    first_ill_cond = ill_cond_size['size'].min()
                    analysis.append(f"Becomes ill-conditioned at size n = {first_ill_cond}")

        return "\n".join(analysis)

    print_detailed_analysis(all_results)
    return (print_detailed_analysis,)


@app.cell
def _():
    """Import of necessary packages and plotting customization"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import polars as pl
        from numpy.linalg import cond, solve, norm
        from scipy.linalg import hilbert
        missing_packages = False
    except ModuleNotFoundError:
        missing_packages = True

    if not missing_packages:
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            # Use LaTeX rendering
            "text.usetex": True,
            "text.latex.preamble" : r"\usepackage{amsmath,bm}",
            "font.family": "serif",
            "font.serif": ["Palatino","Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "figure.figsize": [6, 4],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "mathtext.fontset": "cm",  
        })
    return (
        alt,
        cond,
        hilbert,
        missing_packages,
        norm,
        np,
        pl,
        plt,
        sns,
        solve,
    )


@app.cell
def _(missing_packages, mo):
    module_not_found_explainer = mo.md(
        """
        ## Missing packages warning!

        It looks like you're missing some packages required that this notebook 
        requires.
        Close marimo and check that you have the following packages installed in your virtual environment:
        ```shell
        (.venv) pip install numpy scipy matplotlib seaborn polars
        ```
        """
    ).callout(kind='warn')

    def check_dependencies():
        if missing_packages:
            return module_not_found_explainer
    return check_dependencies, module_not_found_explainer


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
