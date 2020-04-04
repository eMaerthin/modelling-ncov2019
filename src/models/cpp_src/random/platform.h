#pragma once

// Note: __GNUC__ is defined by clang and gcc
#ifdef __GNUC__
#define MOCOS_IMPOSSIBLE(condition) if(condition) __builtin_unreachable();
#define MOCOS_LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#define MOCOS_UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
// For aggressive inlining
#define MOCOS_FORCE_INLINE __attribute__ ((always_inline)) inline
#elif defined _MSC_VER
#define MOCOS_IMPOSSIBLE(condition) __assume(!condition);
#define MOCOS_LIKELY(condition) condition
#define MOCOS_UNLIKELY(condition) condition
#define MOCOS_FORCE_INLINE __forceinline inline
#else
#define MOCOS_IMPOSSIBLE(condition)
#define MOCOS_LIKELY(condition) condition
#define MOCOS_UNLIKELY(condition) condition
#define MOCOS_FORCE_INLINE inline
#endif


#ifdef MOCOS_CPPYY
#define MOCOS_HIDE_IMPLEMENTATIONS true
#else
#define MOCOS_HIDE_IMPLEMENTATIONS false
#endif
