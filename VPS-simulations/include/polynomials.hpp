/**
 * A lightweight implementation of univariate polynomials.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/20/2026
 */

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>

using namespace Eigen;

using boost::multiprecision::abs; 
using boost::multiprecision::sin;
using boost::multiprecision::cos;
using boost::multiprecision::real;

template <size_t N>
using RealType = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N> >; 

template <size_t N>
using ComplexType = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N> >; 

/**
 * A simple univariate polynomial class with complex-valued coefficients.
 *
 * This class assumes Boost.Multiprecision coefficient types. 
 */
template <size_t N>
class HighPrecisionPolynomial
{
    private:
        // Degree of the polynomial
        int degree;

        // Vector of coefficients in increasing order of degree
        Matrix<ComplexType<N>, Dynamic, 1> coefs; 

    public:
        /**
         * Trivial constructor for the zero polynomial. 
         */
        HighPrecisionPolynomial()
        {
            this->degree = 0;
            this->coefs = Matrix<ComplexType<N>, Dynamic, 1>::Zero(1);
        }

        /**
         * Constructor with input real-valued coefficients.
         *
         * @param coefs Input coefficients. 
         */
        HighPrecisionPolynomial(const Ref<const Matrix<RealType<N>, Dynamic, 1> >& coefs)
        {
            // Identify the degree from the highest-degree nonzero coefficient
            this->degree = coefs.size() - 1;
            while (coefs(this->degree) == static_cast<RealType<N> >(0) && this->degree > 0)
                this->degree--;
            
            this->coefs = coefs.head(this->degree + 1).template cast<ComplexType<N> >();
        }

        /**
         * Constructor with input complex-valued coefficients.
         *
         * @param coefs Input coefficients.
         */
        HighPrecisionPolynomial(const Ref<const Matrix<ComplexType<N>, Dynamic, 1> >& coefs)
        {
            // Identify the degree from the highest-degree nonzero coefficient
            this->degree = coefs.size() - 1;
            while (coefs(this->degree) == static_cast<RealType<N> >(0) && this->degree > 0)
                this->degree--;
            
            this->coefs = coefs.head(this->degree + 1);
        }

        /**
         * Trivial destructor.
         */
        ~HighPrecisionPolynomial()
        {
        }

        /**
         * Return the coefficients of the polynomial.
         */
        Matrix<ComplexType<N>, Dynamic, 1> getCoefs() const
        {
            return this->coefs;
        }

        /**
         * Return the degree of the polynomial. 
         */
        int getDegree() const
        {
            return this->degree; 
        }

        /**
         * Evaluate the polynomial at the given complex value with Horner's
         * method.
         *
         * @param x Input value for variable. 
         * @returns Polynomial value. 
         */
        ComplexType<N> eval(const ComplexType<N> x) const
        {
            ComplexType<N> y = this->coefs(this->degree);
            for (int i = this->degree - 1; i >= 0; i--)
                y = this->coefs(i) + x * y;

            return y;
        }

        /**
         * Get the derivative of the polynomial.
         *
         * @returns Derivative polynomial. 
         */
        HighPrecisionPolynomial<N> deriv() const
        {
            // If the polynomial has degree zero, return the zero polynomial
            if (this->degree == 0)
            {
                return HighPrecisionPolynomial<N>();
            }
            // Otherwise, differentiate the polynomial term-by-term
            else
            {
                Matrix<ComplexType<N>, Dynamic, 1> dcoefs(this->degree);
                for (int i = 1; i <= this->degree; ++i)
                    dcoefs(i - 1) = this->coefs(i) * static_cast<RealType<N> >(i);
                return HighPrecisionPolynomial<N>(dcoefs);
            }
        }

        /**
         * Solve for the roots of the polynomial by computing the eigenvalues
         * of the corresponding companion matrix.
         *
         * @param real If true, the coefficients are assumed to be real.
         * @returns Vector of polynomial roots.  
         */
        Matrix<ComplexType<N>, Dynamic, 1> solveCompanion(const bool is_real = false) const
        {
            if (is_real)
            {
                // Define the companion matrix
                Matrix<RealType<N>, Dynamic, Dynamic> companion
                    = Matrix<RealType<N>, Dynamic, Dynamic>::Zero(this->degree, this->degree);
                companion(Eigen::seq(1, this->degree - 1), Eigen::seq(0, this->degree - 2))
                    = Matrix<RealType<N>, Dynamic, Dynamic>::Identity(this->degree - 1, this->degree - 1);
                RealType<N> lead = real(this->coefs(this->degree));
                for (int i = 0; i < this->degree; ++i)
                    companion(i, this->degree - 1) = real(-this->coefs(i)) / lead;

                // Get the eigenvalues of the companion matrix
                EigenSolver<Matrix<RealType<N>, Dynamic, Dynamic> > es; 
                es.compute(companion, false);

                return es.eigenvalues();
            }
            else
            {
                // Define the companion matrix
                Matrix<ComplexType<N>, Dynamic, Dynamic> companion
                    = Matrix<ComplexType<N>, Dynamic, Dynamic>::Zero(this->degree, this->degree);
                companion(Eigen::seq(1, this->degree - 1), Eigen::seq(0, this->degree - 2))
                    = Matrix<ComplexType<N>, Dynamic, Dynamic>::Identity(this->degree - 1, this->degree - 1);
                ComplexType<N> lead = this->coefs(this->degree);
                companion(Eigen::all, this->degree - 1) = -this->coefs.head(this->degree) / lead;

                // Get the eigenvalues of the companion matrix
                ComplexEigenSolver<Matrix<ComplexType<N>, Dynamic, Dynamic> > es; 
                es.compute(companion, false);

                return es.eigenvalues();
            }
        }

        /**
         * Solve for the roots of the polynomial with the Durand-Kerner method.
         *
         * @param tol Tolerance for judging whether the difference between 
         *            consecutive root approximations is zero.
         * @returns Vector of polynomial roots. 
         */
        Matrix<ComplexType<N>, Dynamic, 1> solveDurandKerner(const RealType<N> tol) const
        {
            // Initialize the roots as the n-th roots of unity, where 
            // n is the degree of the polynomial 
            Matrix<ComplexType<N>, Dynamic, 1> roots(this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                RealType<N> a = cos(
                    i * boost::math::constants::two_pi<RealType<N> >() / this->degree
                );
                RealType<N> b = sin(
                    i * boost::math::constants::two_pi<RealType<N> >() / this->degree
                );
                roots(i) = ComplexType<N>(a, b);
            }

            // Store the differences between the roots in a matrix
            Matrix<ComplexType<N>, Dynamic, Dynamic> diffs
                = Matrix<ComplexType<N>, Dynamic, Dynamic>::Zero(this->degree, this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                for (int j = i + 1; j < this->degree; ++j)
                {
                    diffs(i, j) = roots(i) - roots(j); 
                    diffs(j, i) = -diffs(i, j);
                }
            }

            // Iteratively apply the Durand-Kerner method
            RealType<N> max_update = std::numeric_limits<RealType<N> >::infinity();
            while (max_update > tol)
            {
                Matrix<ComplexType<N>, Dynamic, 1> updates(this->degree);

                // Compute the Durand-Kerner update for each root
                for (int i = 0; i < this->degree; ++i)
                {
                    // Evaluate the polynomial at the current value of the
                    // i-th root
                    ComplexType<N> val_i = this->eval(roots(i)); 

                    // Get the product of the differences between the i-th
                    // root and every other root
                    ComplexType<N> dprod = 1.0;
                    for (int j = 0; j < this->degree; ++j)
                    {
                        if (j != i)
                            dprod *= diffs(i, j);
                    }

                    // Compute the Durand-Kerner update
                    updates(i) = val_i / (this->coefs(this->degree) * dprod);
                }

                // Identify the update with the largest magnitude
                max_update = 0.0; 
                for (int i = 0; i < this->degree; ++i)
                {
                    if (max_update < abs(updates(i)))
                        max_update = abs(updates(i)); 
                }

                // Update the roots 
                roots -= updates;

                // Update the pairwise root differences
                for (int i = 0; i < this->degree; ++i)
                {
                    for (int j = i + 1; j < this->degree; ++j)
                    {
                        diffs(i, j) = roots(i) - roots(j); 
                        diffs(j, i) = -diffs(i, j);
                    }
                }
            }

            return roots;
        }

        /**
         * Solve for the roots of the polynomial with Aberth's method.
         *
         * @param tol Tolerance for judging whether the difference between 
         *            consecutive root approximations is zero.
         * @returns Vector of polynomial roots. 
         */
        Matrix<ComplexType<N>, Dynamic, 1> solveAberth(const RealType<N> tol) const
        {
            // Get the derivative of the polynomial 
            HighPrecisionPolynomial<N> deriv = this->deriv();

            // Initialize the roots as the n-th roots of unity, where 
            // n is the degree of the polynomial 
            Matrix<ComplexType<N>, Dynamic, 1> roots(this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                RealType<N> a = cos(
                    i * boost::math::constants::two_pi<RealType<N> >() / this->degree
                );
                RealType<N> b = sin(
                    i * boost::math::constants::two_pi<RealType<N> >() / this->degree
                );
                roots(i) = ComplexType<N>(a, b);
            }

            // Store the differences between the roots in a matrix
            Matrix<ComplexType<N>, Dynamic, Dynamic> diffs
                = Matrix<ComplexType<N>, Dynamic, Dynamic>::Zero(this->degree, this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                for (int j = i + 1; j < this->degree; ++j)
                {
                    diffs(i, j) = roots(i) - roots(j); 
                    diffs(j, i) = -diffs(i, j);
                }
            }

            // Iteratively apply Aberth's method
            RealType<N> max_update = std::numeric_limits<RealType>::infinity();
            while (max_update > tol)
            {
                Matrix<ComplexType<N>, Dynamic, 1> updates(this->degree);

                // Compute the Aberth update for each root 
                for (int i = 0; i < this->degree; ++i)
                {
                    // Evaluate the polynomial and its derivative at the 
                    // current value of the i-th root
                    ComplexType<N> val_i = this->eval(roots(i)); 
                    ComplexType<N> dval_i = deriv.eval(roots(i));
                    ComplexType<N> ratio_i = val_i / dval_i;

                    // Get the sum of one over the difference between the i-th
                    // root and every other root
                    ComplexType<N> dsum = 0;
                    for (int j = 0; j < this->degree; ++j)
                    {
                        if (j != i)
                            dsum += 1.0 / diffs(i, j);
                    }

                    // Compute the Aberth update 
                    updates(i) = ratio_i / (1.0 - ratio_i * dsum);
                }

                // Identify the update with the largest magnitude 
                max_update = 0.0; 
                for (int i = 0; i < this->degree; ++i)
                {
                    if (max_update < abs(updates(i)))
                        max_update = abs(updates(i)); 
                }

                // Update the roots 
                roots -= updates;

                // Update the pairwise root differences
                for (int i = 0; i < this->degree; ++i)
                {
                    for (int j = i + 1; j < this->degree; ++j)
                    {
                        diffs(i, j) = roots(i) - roots(j); 
                        diffs(j, i) = -diffs(i, j);
                    }
                }
            }

            return roots;
        }
};

/**
 * Get the resultant of two univariate polynomials, via the Sylvester matrix.
 *
 * The coefficients of the two polynomials are assumed to be real. 
 *
 * @param f First polynomial. 
 * @param g Second polynomial.
 * @returns Resultant of f and g.  
 */
template <size_t N>
RealType<N> resultant(HighPrecisionPolynomial<N>& f, HighPrecisionPolynomial<N>& g)
{
    // Define the Sylvester matrix
    const int deg_f = f.getDegree(); 
    const int deg_g = g.getDegree();
    Matrix<ComplexType<N>, Dynamic, 1> coefs_f = f.getCoefs(); 
    Matrix<ComplexType<N>, Dynamic, 1> coefs_g = g.getCoefs();  
    Matrix<RealType<N>, Dynamic, Dynamic> sylvester
        = Matrix<RealType<N>, Dynamic, Dynamic>::Zero(deg_f + deg_g, deg_f + deg_g);
    for (int i = 0; i < deg_f; ++i)
    {
        for (int j = 0; j < deg_g; ++j)
        {
            sylvester(i + j, j) = real(coefs_f(i));
        }
    }
    for (int i = 0; i < deg_g; ++i)
    {
        for (int j = 0; j < deg_f; ++j)
        {
            sylvester(i + j, deg_g + j) = real(coefs_g(i));  
        }
    }
   
    // Evaluate the determinant
    return sylvester.determinant();  
}

/**
 * Given a set of points, obtain the unique minimum-degree polynomial that 
 * interpolates these points, via the Vandermonde matrix. 
 *
 * @param points Array of points. 
 * @returns Interpolating polynomial. 
 */
template <size_t N>
HighPrecisionPolynomial<N> interpolant(const Ref<const Array<RealType<N>, Dynamic, 2> >& points)
{
    // Define the Vandermonde matrix 
    const int n = points.rows() - 1;
    Matrix<RealType<N>, Dynamic, Dynamic> A(n + 1, n + 1); 
    A.col(0) = Matrix<RealType<N>, Dynamic, 1>::Ones(n + 1);  
    for (int i = 0; i < n + 1; ++i)
    {
        for (int j = 1; j < n + 1; ++j)
        {
            A(i, j) = pow(points(i, 0), j); 
        }
    }

    // Solve for the interpolating polynomial's coefficients 
    Matrix<RealType<N>, Dynamic, 1> b(n + 1); 
    for (int i = 0; i < n + 1; ++i)
        b(i) = points(i, 1);
    Matrix<RealType<N>, Dynamic, 1> coefs = A.fullPivLu().solve(b);

    return HighPrecisionPolynomial<N>(coefs);  
}

#endif
