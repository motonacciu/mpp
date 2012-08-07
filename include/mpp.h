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

#include "type_traits.h"

#include <vector>
#include <list>
#include <memory>

#include <stdexcept>
#include <sstream>

#include <algorithm>
#include <cassert>
#include <array>

namespace mpi {

// Expection which is thrown every time a communication fails
struct comm_error : public std::logic_error {

	comm_error(const std::string& msg) : std::logic_error(msg) { }

};

class comm;
class endpoint;

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
	msg_impl(const msg_impl<value_type>& other);
	msg_impl<value_type> operator=(const msg_impl<value_type>& other);

	value_type&  m_data;
	int 		 m_tag;
};


template <class T>
inline msg_impl<T> msg(T& raw, int tag=0) { return msg_impl<T>(raw, tag); }

class status;

template <class T>
class request;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// comm: is the abstraction of the MPI_Comm class
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class comm {

	MPI_Comm m_comm;
	bool 	 m_initialized;
	int 	 m_comm_size;
	int 	 m_rank;

	comm(MPI_Comm comm): 
		m_comm(comm), 
		m_initialized(false), 
		m_comm_size(-1), 
		m_rank(-1) { }

	// Check whether MPI_Init has been called
	inline void check_init() {

		if (m_initialized) { return; }

		int flag;
		MPI_Initialized(&flag);
		assert(flag != 0 && 
			"FATAL: MPI environment not initialized (MPI_Init not called)");

		m_initialized = true;
		MPI_Comm_size(m_comm, &m_comm_size);
		MPI_Comm_rank(m_comm, &m_rank);
	}

public:
	// MPI_COMM_WORLD
	static comm world;

	inline int rank() {
		check_init();
		return m_rank;
	}

	inline int rank() const {
		assert(m_initialized && "MPI communicator not initialized");
		return m_rank;
	}

	inline int size() {
		check_init();
		return m_comm_size;
	}

	inline int size() const {
		assert(m_initialized && "MPI communicator not initialized");
		return m_comm_size;
	}

	inline endpoint operator()( const int& rank_id );

};

comm comm::world = comm(MPI_COMM_WORLD);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// endpoint: represent the src or dest of an MPI channel. Provides streaming
// operations to send/recv messages (msg<T>) both in a synchronous or asynch
// ronous way.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class endpoint {

	const int 			m_rank;	 // The rank of this endpoint
	const MPI_Comm& 	m_comm;  // The MPI communicator this endpoing
								 // belongs to

	// Make this class non-copyable 
	endpoint(const endpoint& other);
	endpoint& operator=(const endpoint& other);

public:
	endpoint(const int& rank, const MPI_Comm& com):
		m_rank(rank), m_comm(com) { }

	endpoint(endpoint&& other) :
		m_rank(other.m_rank), 
		m_comm(other.m_comm) { }

	// Send a generic message to this endpoint (synchronously)
	template <class MsgType>
	inline endpoint& operator<<(msg_impl<MsgType>&& m) {
		MPI_Datatype&& dt = m.type();
		if ( MPI_Send( const_cast<void*>(static_cast<const void*>(m.addr())), 
					   static_cast<int>(m.size()), dt,
					   m_rank, 
					   m.tag(), 
					   m_comm
					 ) == MPI_SUCCESS ) {
			return *this;
		}
		std::ostringstream ss;
		ss << "ERROR in MPI rank '" << comm::world.rank()
		   << "': Failed to send message to destination rank '"
		   << m_rank << "'";
		throw comm_error( ss.str() );
	}

	template <class MsgType>
	inline endpoint& operator<<(const msg_impl<MsgType>& m) {
		return operator<<(std::move(m));
	}

	template <class RawType>
	inline endpoint& operator<<(const RawType& m) {
		return operator<<( std::move( msg_impl<const RawType>(m) ) );
	}

	// Receive from this endpoint (synchronously)
	template <class RawType>
	inline status operator>>(RawType& m);

	template <class MsgType>
	inline status operator>>(msg_impl<MsgType>&& m);

	// Receive from this endpoing (asynchronously)
	template <class MsgType>
	inline request<MsgType> operator>(msg_impl<MsgType>&& m) {
		MPI_Request req;
		if( MPI_Irecv( static_cast<void*>(m.addr()), 
					   static_cast<int>(m.size()), m.type(),
					   m_rank, m.tag(), m_comm, &req
					 ) != MPI_SUCCESS ) {
			std::ostringstream ss;
			ss << "ERROR in MPI rank '" << comm::world.rank()
			   << "': Failed to receive message from destination rank '"
			   << m_rank << "'";
			throw comm_error( ss.str() );
		}
		return request<MsgType>(m_comm, req, std::move(m));
	}

	// Receive from this endpoing (asynchronously
	template <class RawType>
	inline request<RawType> operator>(RawType& m) {
		return operator>( std::move(msg_impl<RawType>(m)) );
	}

	// Returns the rank of this endpoit
	inline const int& rank() const { return m_rank; }
};

inline endpoint comm::operator()(const int& rank_id) {
	return endpoint(rank_id, m_comm);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// status: keeps the status info for received messages.
// containes the sender of the message, tag and size of
// the message
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class status{

	const MPI_Comm&      m_comm;
    const MPI_Status     m_status;
	const MPI_Datatype   m_datatype;

public:
	status(const MPI_Comm& com, const MPI_Status& s, const MPI_Datatype& dt):
		m_comm(com), m_status(s), m_datatype(dt) { }

	inline endpoint source(){
		return endpoint(m_status.MPI_SOURCE, m_comm);
	}

	inline int count(){
		int count;
		MPI_Get_count(const_cast<MPI_Status*>(&m_status), m_datatype, &count);
		return count;
	}

	inline int tag(){
		return m_status.MPI_TAG;
	}

	inline int error(){
		return m_status.MPI_ERROR;
	}
};

template <class RawType>
inline status endpoint::operator>>(RawType& m) {
	return operator>>( msg_impl<RawType>( m ) );
}

template <class MsgType>
inline status endpoint::operator>>(msg_impl<MsgType>&& m) {
	MPI_Status s;
	MPI_Datatype dt = m.type();
	if(MPI_Recv( const_cast<void*>(static_cast<const void*>(m.addr())), 
				 static_cast<int>(m.size()), dt,
				 m_rank, m.tag(), m_comm, &s
			   ) == MPI_SUCCESS ) {
		return status(m_comm, s, dt);
	}
	std::ostringstream ss;
	ss << "ERROR in MPI rank '" << comm::world.rank()
	   << "': Failed to receive message from destination rank '"
	   << m_rank << "'";
	throw comm_error( ss.str() );
	//sint size;
	//MPI_Get_count(&s, MPI_BYTE, &size);
	//std::cout << "RECEIVED " << size << std::endl;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// request<T> is an implementation of the future concept used for asynchronous
// receives/sends (TODO)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class T>
class request{
	MPI_Comm const&     		m_comm;
	MPI_Request 				m_req;
	msg_impl<T>					m_msg;
	std::unique_ptr<status> 	m_status;
	int 		 				done;

public:
	request(MPI_Comm const& com, MPI_Request req, msg_impl<T>&& msg):
		m_comm(com), m_req(req), m_msg(std::move(msg)), done(0) { }

	request(request<T>&& other) : 
		m_comm( std::move(other.m_comm) ), 
		m_req( std::move(other.m_req) ), 
		m_msg( std::move(other.m_msg) ),
		m_status( std::move(other.m_status) ),
		done(other.done) { }

	void cancel();

	inline const T& get() {
		if ( !done ) {
			MPI_Status stat;
			// wait to receive the message
			MPI_Wait(&m_req, &stat);
			done = 1;
			m_status = std::unique_ptr<status>( new status(m_comm, stat, m_msg.type()) );
		}
		return m_msg.get();
	}

	inline status getStatus() {
		if( isDone() ) { return *m_status; }
		throw "not done";
	}

	inline bool isDone() {
		if ( !done ) {
			MPI_Status stat;
			MPI_Test(&m_req, &done, &stat);
			if ( done ) {
				m_status = std::unique_ptr<status>( new status(m_comm, stat, m_msg.type()) );
			}
		}
		return done;
	}
};

const int any = MPI_ANY_SOURCE;

inline void init(int argc = 0, char* argv[] = NULL){ MPI_Init(&argc, &argv); }
inline void finalize(){ MPI_Finalize(); }

} // end mpi namespace

