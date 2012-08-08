/******************************************************************************
 *
 * 					 	 	MPP: An MPI CPP Interface
 *
 * 					Copyright (C) 2011-2012  Simone Pellegrini
 *
 * 	This library is free software; you can redistribute it and/or modify it
 * 	under the terms of the GNU Lesser General Public License as published by the
 * 	Free Software Foundation; either version 2.1 of the License, or (at your
 * 	option) any later version.
 *
 * 	This library is distributed in the hope that it will be useful, but WITHOUT
 * 	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * 	FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * 	for more details.
 *
 * 	You should have received a copy of the GNU Lesser General Public License
 * 	along with this library; if not, write to the Free Software Foundation,
 * 	Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 ******************************************************************************/

#pragma once

#include <mpi.h>

#include <list>
#include <array>
#include <algorithm>
#include <complex>

namespace mpi {

//*****************************************************************************
// 									MPI Type Traits
//*****************************************************************************
template <class T>
struct mpi_type_traits {

	typedef T element_type;
	typedef T* element_addr_type;

	static inline MPI_Datatype get_type(T&& raw);

	static inline size_t get_size(T& raw) { return 1; }

	static inline element_addr_type get_addr(T& raw) { return &raw; }

};

/** 
 * Specialization of the mpi_type_traits for primitive types 
 */
#define PRIMITIVE(Type, MpiType) \
	template<> \
	inline MPI_Datatype mpi_type_traits<Type>::get_type(Type&&) { \
		return MpiType; \
	}

PRIMITIVE(char, 				MPI::CHAR);
PRIMITIVE(wchar_t,				MPI::WCHAR);
PRIMITIVE(short, 				MPI::SHORT);
PRIMITIVE(int, 					MPI::INT);
PRIMITIVE(long, 				MPI::LONG);
PRIMITIVE(signed char, 			MPI::SIGNED_CHAR);
PRIMITIVE(unsigned char, 		MPI::UNSIGNED_CHAR);
PRIMITIVE(unsigned short, 		MPI::UNSIGNED_SHORT);
PRIMITIVE(unsigned int,			MPI::UNSIGNED);
PRIMITIVE(unsigned long,		MPI::UNSIGNED_LONG);
PRIMITIVE(unsigned long long,	MPI::UNSIGNED_LONG_LONG);

PRIMITIVE(float, 				MPI::FLOAT);
PRIMITIVE(double, 				MPI::DOUBLE);
PRIMITIVE(long double,			MPI::LONG_DOUBLE);

PRIMITIVE(bool,						MPI::BOOL);
PRIMITIVE(std::complex<float>,		MPI::COMPLEX);
PRIMITIVE(std::complex<double>,		MPI::DOUBLE_COMPLEX);
PRIMITIVE(std::complex<long double>,	MPI::LONG_DOUBLE_COMPLEX);

#undef PRIMITIVE 

// ... add missing types here ...

template <class T>
struct mpi_type_traits<const T> {
	
	typedef const typename mpi_type_traits<T>::element_type element_type;
	typedef const typename mpi_type_traits<T>::element_addr_type element_addr_type;

	static inline size_t get_size(const T& elem) {
		return mpi_type_traits<T>::get_size( const_cast<T&>(elem) );
	}

	static inline MPI_Datatype get_type(const T& elem) {
		return mpi_type_traits<T>::get_type( T() );
	}

	static inline element_addr_type get_addr(const T& elem) {
		return mpi_type_traits<T>::get_addr( const_cast<T&>(elem) );
	}
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	std::vector<T> traits
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class T>
struct mpi_type_traits<std::vector<T>> {
	
	typedef T element_type;
	typedef T* element_addr_type;

	static inline size_t get_size(std::vector<T>& vec) { return vec.size(); }

	static inline MPI_Datatype get_type(std::vector<T>&& vec) {
		return mpi_type_traits<T>::get_type( T() );
	}

	static inline element_addr_type get_addr(std::vector<T>& vec) {
		return  mpi_type_traits<T>::get_addr( vec.front() );
	}

};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	std::array<T> traits
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class T, size_t N>
struct mpi_type_traits<std::array<T,N>> {

	inline static size_t get_size(const std::array<T,N>& vec) { return N; }

	inline static MPI_Datatype get_type(const std::array<T,N>& vec) {
		return  mpi_type_traits<T>::get_type( T() );
	}

	static inline const T* get_addr(const std::array<T,N>& vec) {
		return mpi_type_traits<T>::get_addr( vec.front() );
	}
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	std::list<T> traits
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class T>
struct mpi_type_traits<std::list<T>> {

	static inline size_t get_size(const std::list<T>& vec) { return 1; }

	static MPI_Datatype get_type(const std::list<T>& l) {
		// we have to get the create an MPI_Datatype containing the offsets
		// of the current object

		// we consider the offsets starting from the first element
		std::vector<MPI_Aint> address( l.size() );
		std::vector<int> dimension( l.size() );
		std::vector<MPI_Datatype> types( l.size() );

		std::vector<int>::iterator dim_it = dimension.begin();
		std::vector<MPI_Aint>::iterator address_it = address.begin();
		std::vector<MPI_Datatype>::iterator type_it = types.begin();

		MPI_Aint base_address;
		MPI_Address(const_cast<T*>(&l.front()), &base_address);

		*(type_it++) = mpi_type_traits<T>::get_type( l.front() );
		*(dim_it++) = static_cast<int>(mpi_type_traits<T>::get_size( l.front() ));
		*(address_it++) = 0;

		typename std::list<T>::const_iterator begin = l.begin();
		++begin;
		std::for_each(begin, l.cend(), [&](const T& curr) {
				assert( address_it != address.end() &&
						  type_it != types.end() &&
						  dim_it != dimension.end() );

				MPI_Address(const_cast<T*>(&curr), &*address_it);
				*(address_it++) -= base_address;
				*(type_it++) =  mpi_type_traits<T>::get_type( curr );
				*(dim_it++) = static_cast<int>(mpi_type_traits<T>::get_size( curr ));

			}
		);

		MPI_Datatype list_dt;
		MPI_Type_create_struct(static_cast<int>(l.size()), &dimension.front(), &address.front(), &types.front(), &list_dt);
		MPI_Type_commit( &list_dt );

		return list_dt;
	}

	static inline const T* get_addr(const std::list<T>& list) {
		return  mpi_type_traits<T>::get_addr( list.front() );
	}

};

} // end mpi namespace 

