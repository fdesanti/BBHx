#ifndef CUDA_COMPLEX_HPP
#define CUDA_COMPLEX_HPP

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cmath>
#include <sstream>
#include <complex>
#include <limits>

namespace gcmplx {

template<class _Tp> class complex;

template<class _Tp> complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);
template<class _Tp> complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

template<class _Tp>
class complex {
public:
    typedef _Tp value_type;
private:
    value_type __re_;
    value_type __im_;
public:
    CUDA_CALLABLE_MEMBER
    complex(const value_type& __re = value_type(), const value_type& __im = value_type())
        : __re_(__re), __im_(__im) {}
    template<class _Xp> CUDA_CALLABLE_MEMBER
    complex(const complex<_Xp>& __c)
        : __re_(__c.real()), __im_(__c.imag()) {}

    CUDA_CALLABLE_MEMBER value_type real() const { return __re_; }
    CUDA_CALLABLE_MEMBER value_type imag() const { return __im_; }

    CUDA_CALLABLE_MEMBER void real(value_type __re) { __re_ = __re; }
    CUDA_CALLABLE_MEMBER void imag(value_type __im) { __im_ = __im; }

    CUDA_CALLABLE_MEMBER complex& operator=(const value_type& __re)
        { __re_ = __re; __im_ = value_type(); return *this; }
    CUDA_CALLABLE_MEMBER complex& operator+=(const value_type& __re) { __re_ += __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator-=(const value_type& __re) { __re_ -= __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator*=(const value_type& __re) { __re_ *= __re; __im_ *= __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator/=(const value_type& __re) { __re_ /= __re; __im_ /= __re; return *this; }

    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator=(const complex<_Xp>& __c) {
        __re_ = __c.real();
        __im_ = __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator+=(const complex<_Xp>& __c) {
        __re_ += __c.real();
        __im_ += __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator-=(const complex<_Xp>& __c) {
        __re_ -= __c.real();
        __im_ -= __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator*=(const complex<_Xp>& __c) {
        *this = *this * __c;
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator/=(const complex<_Xp>& __c) {
        *this = *this / __c;
        return *this;
    }
};

template<> class complex<double>;

template<>
class complex<float> {
    float __re_;
    float __im_;
public:
    typedef float value_type;
    CUDA_CALLABLE_MEMBER complex(float __re = 0.0f, float __im = 0.0f)
        : __re_(__re), __im_(__im) {}
    explicit CUDA_CALLABLE_MEMBER complex(const complex<double>& __c);

    CUDA_CALLABLE_MEMBER float real() const { return __re_; }
    CUDA_CALLABLE_MEMBER float imag() const { return __im_; }

    CUDA_CALLABLE_MEMBER void real(value_type __re) { __re_ = __re; }
    CUDA_CALLABLE_MEMBER void imag(value_type __im) { __im_ = __im; }

    CUDA_CALLABLE_MEMBER complex& operator=(float __re)
        { __re_ = __re; __im_ = value_type(); return *this; }
    CUDA_CALLABLE_MEMBER complex& operator+=(float __re) { __re_ += __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator-=(float __re) { __re_ -= __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator*=(float __re) { __re_ *= __re; __im_ *= __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator/=(float __re) { __re_ /= __re; __im_ /= __re; return *this; }

    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator=(const complex<_Xp>& __c) {
        __re_ = __c.real();
        __im_ = __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator+=(const complex<_Xp>& __c) {
        __re_ += __c.real();
        __im_ += __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator-=(const complex<_Xp>& __c) {
        __re_ -= __c.real();
        __im_ -= __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator*=(const complex<_Xp>& __c) {
        *this = *this * __c;
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator/=(const complex<_Xp>& __c) {
        *this = *this / __c;
        return *this;
    }
};

template<>
class complex<double> {
    double __re_;
    double __im_;
public:
    typedef double value_type;
    CUDA_CALLABLE_MEMBER complex(double __re = 0.0, double __im = 0.0)
        : __re_(__re), __im_(__im) {}
    CUDA_CALLABLE_MEMBER complex(const complex<float>& __c);

    CUDA_CALLABLE_MEMBER double real() const { return __re_; }
    CUDA_CALLABLE_MEMBER double imag() const { return __im_; }

    CUDA_CALLABLE_MEMBER void real(value_type __re) { __re_ = __re; }
    CUDA_CALLABLE_MEMBER void imag(value_type __im) { __im_ = __im; }

    CUDA_CALLABLE_MEMBER complex& operator=(double __re)
        { __re_ = __re; __im_ = value_type(); return *this; }
    CUDA_CALLABLE_MEMBER complex& operator+=(double __re) { __re_ += __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator-=(double __re) { __re_ -= __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator*=(double __re) { __re_ *= __re; __im_ *= __re; return *this; }
    CUDA_CALLABLE_MEMBER complex& operator/=(double __re) { __re_ /= __re; __im_ /= __re; return *this; }

    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator=(const complex<_Xp>& __c) {
        __re_ = __c.real();
        __im_ = __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator+=(const complex<_Xp>& __c) {
        __re_ += __c.real();
        __im_ += __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator-=(const complex<_Xp>& __c) {
        __re_ -= __c.real();
        __im_ -= __c.imag();
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator*=(const complex<_Xp>& __c) {
        *this = *this * __c;
        return *this;
    }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator/=(const complex<_Xp>& __c) {
        *this = *this / __c;
        return *this;
    }
};

// Specialized constructors
inline CUDA_CALLABLE_MEMBER
complex<float>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline CUDA_CALLABLE_MEMBER
complex<double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

// 26.3.6 operators:

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator+(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator+(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t += __x;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator-(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator-(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(-__y);
    __t += __x;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    _Tp __x = __ac - __bd;
    _Tp __y = __ad + __bc;
    if (std::isnan(__x) && std::isnan(__y))
    {
        bool __recalc = false;
        if (std::isinf(__a) || std::isinf(__b))
        {
            __a = std::copysign(std::isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = std::copysign(std::isinf(__b) ? _Tp(1) : _Tp(0), __b);
            if (std::isnan(__c))
                __c = std::copysign(_Tp(0), __c);
            if (std::isnan(__d))
                __d = std::copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (std::isinf(__c) || std::isinf(__d))
        {
            __c = std::copysign(std::isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = std::copysign(std::isinf(__d) ? _Tp(1) : _Tp(0), __d);
            if (std::isnan(__a))
                __a = std::copysign(_Tp(0), __a);
            if (std::isnan(__b))
                __b = std::copysign(_Tp(0), __b);
            __recalc = true;
        }
        if (!__recalc && (std::isinf(__ac) || std::isinf(__bd) ||
                          std::isinf(__ad) || std::isinf(__bc)))
        {
            if (std::isnan(__a))
                __a = std::copysign(_Tp(0), __a);
            if (std::isnan(__b))
                __b = std::copysign(_Tp(0), __b);
            if (std::isnan(__c))
                __c = std::copysign(_Tp(0), __c);
            if (std::isnan(__d))
                __d = std::copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (__recalc)
        {
            __x = _Tp(INFINITY) * (__a * __c - __b * __d);
            __y = _Tp(INFINITY) * (__a * __d + __b * __c);
        }
    }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator*(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t *= __y;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator*(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t *= __x;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    int __ilogbw = 0;
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __logbw = std::logb(std::fmax(std::fabs(__c), std::fabs(__d)));
    if (std::isfinite(__logbw))
    {
        __ilogbw = static_cast<int>(__logbw);
        __c = std::scalbn(__c, -__ilogbw);
        __d = std::scalbn(__d, -__ilogbw);
    }
    _Tp __denom = __c * __c + __d * __d;
    _Tp __x = std::scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    _Tp __y = std::scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (std::isnan(__x) && std::isnan(__y))
    {
        if ((__denom == _Tp(0)) && (!std::isnan(__a) || !std::isnan(__b)))
        {
            __x = std::copysign(_Tp(INFINITY), __c) * __a;
            __y = std::copysign(_Tp(INFINITY), __c) * __b;
        }
        else if ((std::isinf(__a) || std::isinf(__b)) && std::isfinite(__c) && std::isfinite(__d))
        {
            __a = std::copysign(std::isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = std::copysign(std::isinf(__b) ? _Tp(1) : _Tp(0), __b);
            __x = _Tp(INFINITY) * (__a * __c + __b * __d);
            __y = _Tp(INFINITY) * (__b * __c - __a * __d);
        }
        else if (std::isinf(__logbw) && __logbw > _Tp(0) && std::isfinite(__a) && std::isfinite(__b))
        {
            __c = std::copysign(std::isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = std::copysign(std::isinf(__d) ? _Tp(1) : _Tp(0), __d);
            __x = _Tp(0) * (__a * __c + __b * __d);
            __y = _Tp(0) * (__b * __c - __a * __d);
        }
    }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator/(const complex<_Tp>& __x, const _Tp& __y)
{
    return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator/(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t /= __y;
    return __t;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator+(const complex<_Tp>& __x)
{
    return __x;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> operator-(const complex<_Tp>& __x)
{
    return complex<_Tp>(-__x.real(), -__x.imag());
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
bool operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
bool operator==(const complex<_Tp>& __x, const _Tp& __y)
{
    return __x.real() == __y && __x.imag() == 0;
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
bool operator==(const _Tp& __x, const complex<_Tp>& __y)
{
    return __x == __y.real() && 0 == __y.imag();
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
bool operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
bool operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
bool operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

// 26.3.7 values:

// real

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
_Tp real(const complex<_Tp>& __c)
{
    return __c.real();
}

inline CUDA_CALLABLE_MEMBER
double real(double __re)
{
    return __re;
}

inline CUDA_CALLABLE_MEMBER
float real(float __re)
{
    return __re;
}

// imag

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
_Tp imag(const complex<_Tp>& __c)
{
    return __c.imag();
}

inline CUDA_CALLABLE_MEMBER
double imag(double __re)
{
    return 0;
}

inline CUDA_CALLABLE_MEMBER
float imag(float __re)
{
    return 0;
}

// abs

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
_Tp abs(const complex<_Tp>& __c)
{
    return std::hypot(__c.real(), __c.imag());
}

// arg

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
_Tp arg(const complex<_Tp>& __c)
{
    return std::atan2(__c.imag(), __c.real());
}

inline CUDA_CALLABLE_MEMBER
double arg(double __re)
{
    return std::atan2(0., __re);
}

inline CUDA_CALLABLE_MEMBER
float arg(float __re)
{
    return std::atan2(0.F, __re);
}

// norm

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
_Tp norm(const complex<_Tp>& __c)
{
    if (std::isinf(__c.real()))
        return std::fabs(__c.real());
    if (std::isinf(__c.imag()))
        return std::fabs(__c.imag());
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

inline CUDA_CALLABLE_MEMBER
double norm(double __re)
{
    return __re * __re;
}

inline CUDA_CALLABLE_MEMBER
float norm(float __re)
{
    return __re * __re;
}

// conj

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> conj(const complex<_Tp>& __c)
{
    return complex<_Tp>(__c.real(), -__c.imag());
}

inline CUDA_CALLABLE_MEMBER
complex<double> conj(double __re)
{
    return complex<double>(__re);
}

inline CUDA_CALLABLE_MEMBER
complex<float> conj(float __re)
{
    return complex<float>(__re);
}

// proj

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> proj(const complex<_Tp>& __c)
{
    complex<_Tp> __r = __c;
    if (std::isinf(__c.real()) || std::isinf(__c.imag()))
        __r = complex<_Tp>(std::numeric_limits<_Tp>::infinity(), std::copysign(_Tp(0), __c.imag()));
    return __r;
}

inline CUDA_CALLABLE_MEMBER
complex<double> proj(double __re)
{
    if (std::isinf(__re))
        __re = std::fabs(__re);
    return complex<double>(__re);
}

inline CUDA_CALLABLE_MEMBER
complex<float> proj(float __re)
{
    if (std::isinf(__re))
        __re = std::fabs(__re);
    return complex<float>(__re);
}

// polar

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> polar(const _Tp& __rho, const _Tp& __theta = _Tp(0))
{
    if (std::isnan(__rho) || std::signbit(__rho))
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    if (std::isnan(__theta))
    {
        if (std::isinf(__rho))
            return complex<_Tp>(__rho, __theta);
        return complex<_Tp>(__theta, __theta);
    }
    if (std::isinf(__theta))
    {
        if (std::isinf(__rho))
            return complex<_Tp>(__rho, _Tp(NAN));
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    }
    _Tp __x = __rho * std::cos(__theta);
    if (std::isnan(__x))
        __x = 0;
    _Tp __y = __rho * std::sin(__theta);
    if (std::isnan(__y))
        __y = 0;
    return complex<_Tp>(__x, __y);
}

// log

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> log(const complex<_Tp>& __x)
{
    return complex<_Tp>(std::log(abs(__x)), arg(__x));
}

// log10

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> log10(const complex<_Tp>& __x)
{
    return log(__x) / std::log(_Tp(10));
}

// sqrt

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> sqrt(const complex<_Tp>& __x)
{
    if (std::isinf(__x.imag()))
        return complex<_Tp>(_Tp(std::numeric_limits<_Tp>::infinity()), __x.imag());
    if (std::isinf(__x.real()))
    {
        if (__x.real() > _Tp(0))
            return complex<_Tp>(__x.real(), std::isnan(__x.imag()) ? __x.imag() : std::copysign(_Tp(0), __x.imag()));
        return complex<_Tp>(std::isnan(__x.imag()) ? __x.imag() : _Tp(0), std::copysign(__x.real(), __x.imag()));
    }
    return polar(std::sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> exp(const complex<_Tp>& __x)
{
    _Tp __i = __x.imag();
    if (std::isinf(__x.real()))
    {
        if (__x.real() < _Tp(0))
        {
            if (!std::isfinite(__i))
                __i = _Tp(1);
        }
        else if (__i == 0 || !std::isfinite(__i))
        {
            if (std::isinf(__i))
                __i = _Tp(NAN);
            return complex<_Tp>(__x.real(), __i);
        }
    }
    else if (std::isnan(__x.real()) && __x.imag() == 0)
        return __x;
    _Tp __e = std::exp(__x.real());
    return complex<_Tp>(__e * std::cos(__i), __e * std::sin(__i));
}

// pow

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return exp(__y * log(__x));
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> pow(const complex<_Tp>& __x, const _Tp& __y)
{
    return pow(__x, complex<_Tp>(__y));
}

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> pow(const _Tp& __x, const complex<_Tp>& __y)
{
    return pow(complex<_Tp>(__x), __y);
}

// asinh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> asinh(const complex<_Tp>& __x)
{
    const _Tp __pi(std::atan2(+0., -0.));
    if (std::isinf(__x.real()))
    {
        if (std::isnan(__x.imag()))
            return __x;
        if (std::isinf(__x.imag()))
            return complex<_Tp>(__x.real(), std::copysign(__pi * _Tp(0.25), __x.imag()));
        return complex<_Tp>(__x.real(), std::copysign(_Tp(0), __x.imag()));
    }
    if (std::isnan(__x.real()))
    {
        if (std::isinf(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (__x.imag() == 0)
            return __x;
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (std::isinf(__x.imag()))
        return complex<_Tp>(std::copysign(__x.imag(), __x.real()), std::copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(pow(__x, _Tp(2)) + _Tp(1)));
    return complex<_Tp>(std::copysign(__z.real(), __x.real()), std::copysign(__z.imag(), __x.imag()));
}

// acosh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> acosh(const complex<_Tp>& __x)
{
    const _Tp __pi(std::atan2(+0., -0.));
    if (std::isinf(__x.real()))
    {
        if (std::isnan(__x.imag()))
            return complex<_Tp>(std::fabs(__x.real()), __x.imag());
        if (std::isinf(__x.imag()))
            if (__x.real() > 0)
                return complex<_Tp>(__x.real(), std::copysign(__pi * _Tp(0.25), __x.imag()));
            else
                return complex<_Tp>(-__x.real(), std::copysign(__pi * _Tp(0.75), __x.imag()));
        if (__x.real() < 0)
            return complex<_Tp>(-__x.real(), std::copysign(__pi, __x.imag()));
        return complex<_Tp>(__x.real(), std::copysign(_Tp(0), __x.imag()));
    }
    if (std::isnan(__x.real()))
    {
        if (std::isinf(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (std::isinf(__x.imag()))
        return complex<_Tp>(std::fabs(__x.imag()), std::copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(pow(__x, _Tp(2)) - _Tp(1)));
    return complex<_Tp>(std::copysign(__z.real(), _Tp(0)), std::copysign(__z.imag(), __x.imag()));
}

// atanh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> atanh(const complex<_Tp>& __x)
{
    const _Tp __pi(std::atan2(+0., -0.));
    if (std::isinf(__x.imag()))
    {
        return complex<_Tp>(std::copysign(_Tp(0), __x.real()), std::copysign(__pi/_Tp(2), __x.imag()));
    }
    if (std::isnan(__x.imag()))
    {
        if (std::isinf(__x.real()) || __x.real() == 0)
            return complex<_Tp>(std::copysign(_Tp(0), __x.real()), __x.imag());
        return complex<_Tp>(__x.imag(), __x.imag());
    }
    if (std::isnan(__x.real()))
    {
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (std::isinf(__x.real()))
    {
        return complex<_Tp>(std::copysign(_Tp(0), __x.real()), std::copysign(__pi/_Tp(2), __x.imag()));
    }
    if (std::fabs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0))
    {
        return complex<_Tp>(std::copysign(_Tp(INFINITY), __x.real()), std::copysign(_Tp(0), __x.imag()));
    }
    complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
    return complex<_Tp>(std::copysign(__z.real(), __x.real()), std::copysign(__z.imag(), __x.imag()));
}

// sinh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> sinh(const complex<_Tp>& __x)
{
    if (std::isinf(__x.real()) && !std::isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.real() == 0 && !std::isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.imag() == 0 && !std::isfinite(__x.real()))
        return __x;
    return complex<_Tp>(std::sinh(__x.real()) * std::cos(__x.imag()),
                         std::cosh(__x.real()) * std::sin(__x.imag()));
}

// cosh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> cosh(const complex<_Tp>& __x)
{
    if (std::isinf(__x.real()) && !std::isfinite(__x.imag()))
        return complex<_Tp>(std::fabs(__x.real()), _Tp(NAN));
    if (__x.real() == 0 && !std::isfinite(__x.imag()))
        return complex<_Tp>(_Tp(NAN), __x.real());
    if (__x.real() == 0 && __x.imag() == 0)
        return complex<_Tp>(_Tp(1), __x.imag());
    if (__x.imag() == 0 && !std::isfinite(__x.real()))
        return complex<_Tp>(std::fabs(__x.real()), __x.imag());
    return complex<_Tp>(std::cosh(__x.real()) * std::cos(__x.imag()),
                         std::sinh(__x.real()) * std::sin(__x.imag()));
}

// tanh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> tanh(const complex<_Tp>& __x)
{
    if (std::isinf(__x.real()))
    {
        if (!std::isfinite(__x.imag()))
            return complex<_Tp>(_Tp(1), _Tp(0));
        return complex<_Tp>(_Tp(1), std::copysign(_Tp(0), std::sin(_Tp(2) * __x.imag())));
    }
    if (std::isnan(__x.real()) && __x.imag() == 0)
        return __x;
    _Tp __2r = _Tp(2) * __x.real();
    _Tp __2i = _Tp(2) * __x.imag();
    _Tp __d = std::cosh(__2r) + std::cos(__2i);
    return complex<_Tp>(std::sinh(__2r)/__d, std::sin(__2i)/__d);
}

// asin

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> asin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> acos(const complex<_Tp>& __x)
{
    const _Tp __pi(std::atan2(+0., -0.));
    if (std::isinf(__x.real()))
    {
        if (std::isnan(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (std::isinf(__x.imag()))
        {
            if (__x.real() < _Tp(0))
                return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
            return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
        }
        if (__x.real() < _Tp(0))
            return complex<_Tp>(__pi, std::signbit(__x.imag()) ? -__x.real() : __x.real());
        return complex<_Tp>(_Tp(0), std::signbit(__x.imag()) ? __x.real() : -__x.real());
    }
    if (std::isnan(__x.real()))
    {
        if (std::isinf(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (std::isinf(__x.imag()))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    if (__x.real() == 0)
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    complex<_Tp> __z = log(__x + sqrt(pow(__x, _Tp(2)) - _Tp(1)));
    if (std::signbit(__x.imag()))
        return complex<_Tp>(std::fabs(__z.imag()), std::fabs(__z.real()));
    return complex<_Tp>(std::fabs(__z.imag()), -std::fabs(__z.real()));
}

// atan

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> atan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> sin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template<class _Tp>
inline CUDA_CALLABLE_MEMBER
complex<_Tp> cos(const complex<_Tp>& __x)
{
    return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp> tan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

template<class _Tp, class _CharT, class _Traits>
std::basic_istream<_CharT, _Traits>&
operator>>(std::basic_istream<_CharT, _Traits>& __is, complex<_Tp>& __x)
{
    if (__is.good())
    {
        std::ws(__is);
        if (__is.peek() == _CharT('('))
        {
            __is.get();
            _Tp __r;
            __is >> __r;
            if (!__is.fail())
            {
                std::ws(__is);
                _CharT __c = __is.peek();
                if (__c == _CharT(','))
                {
                    __is.get();
                    _Tp __i;
                    __is >> __i;
                    if (!__is.fail())
                    {
                        std::ws(__is);
                        __c = __is.peek();
                        if (__c == _CharT(')'))
                        {
                            __is.get();
                            __x = complex<_Tp>(__r, __i);
                        }
                        else
                            __is.setstate(std::ios_base::failbit);
                    }
                    else
                        __is.setstate(std::ios_base::failbit);
                }
                else if (__c == _CharT(')'))
                {
                    __is.get();
                    __x = complex<_Tp>(__r, _Tp(0));
                }
                else
                    __is.setstate(std::ios_base::failbit);
            }
            else
                __is.setstate(std::ios_base::failbit);
        }
        else
        {
            _Tp __r;
            __is >> __r;
            if (!__is.fail())
                __x = complex<_Tp>(__r, _Tp(0));
            else
                __is.setstate(std::ios_base::failbit);
        }
    }
    else
        __is.setstate(std::ios_base::failbit);
    return __is;
}

template<class _Tp, class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, const complex<_Tp>& __x)
{
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << '(' << __x.real() << ',' << __x.imag() << ')';
    return __os << __s.str();
}

} // namespace gcmplx

#endif // CUDA_COMPLEX_HPP