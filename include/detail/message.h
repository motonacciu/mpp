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

#include "detail/decls.h"

#include "type_traits.h"

namespace mpi {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// msg: represent a single message which can be provided to the <<, <, >>, >
// operations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class MsgTy>
struct msg_impl {

	typedef MsgTy value_type;

	// Builds a msg wrapping v
	msg_impl(value_type& v, int tag = 0) : m_data(v), m_tag(tag) { }

	// Move copy constructor 
	msg_impl(msg_impl<value_type>&& other) : 
		m_data(other.m_data), 
		m_tag(other.m_tag) { }

	inline typename mpi_type_traits<value_type>::element_addr_type addr() const {
		return mpi_type_traits<value_type>::get_addr(m_data);
	}

	inline const value_type& get() const { return m_data; }

	// Returns the dimension of this message
	inline size_t size() const {
		return mpi_type_traits<value_type>::get_size(m_data);
	}

	inline MPI_Datatype type() const {
		return mpi_type_traits<value_type>::get_type(std::move(m_data));
	}

	// getter/setter for m_tag
	inline const int& tag() const { return m_tag; }
	inline int& tag() { return m_tag; }

private:

	// Make this class non-copyable 
	msg_impl(const msg_impl<value_type>& other) = delete;
	msg_impl<value_type> operator=(const msg_impl<value_type>& other) = delete;

	value_type&  m_data;
	int 		 m_tag;
};


template <class T>
inline msg_impl<T> msg(T& raw, int tag=0) { return msg_impl<T>(raw, tag); }

} // end mpi namespace 
