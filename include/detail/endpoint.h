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

namespace mpi {

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
	endpoint(const endpoint& other) = delete;
	endpoint& operator=(const endpoint& other) = delete;

public:
	endpoint(const int& rank, const MPI_Comm& com):
		m_rank(rank), m_comm(com) { }

	endpoint(endpoint&& other) :
		m_rank(other.m_rank), 
		m_comm(other.m_comm) { }


	// Send a generic message to this endpoint (synchronously)
	template <class MsgType> 
	inline endpoint& send(msg_impl<MsgType>&& m);

	template <class MsgType>
	inline endpoint& send(const msg_impl<MsgType>& m) {
		return send(std::move(m));
	}

	template <class RawType>
	inline endpoint& send(const RawType& m) {
		return send( std::move( msg_impl<const RawType>(m) ) );
	}

	template <class MsgType>
	inline endpoint& operator<<(msg_impl<MsgType>&& m) { 
		return send(std::move(m)); 
	}

	template <class MsgType>
	inline endpoint& operator<<(const msg_impl<MsgType>& m) {
		return send(std::move(m));
	}

	template <class RawType>
	inline endpoint& operator<<(const RawType& m) {
		return send( std::move( msg_impl<const RawType>(m) ) );
	}


	// Receive from this endpoint (synchronously)
	template <class RawType>
	inline status operator>>(RawType& m);

	template <class MsgType>
	inline status operator>>(msg_impl<MsgType>&& m);

	// Receive from this endpoing (asynchronously)
	template <class MsgType>
	inline request<MsgType> operator>(msg_impl<MsgType>&& m);

	// Receive from this endpoing (asynchronously
	template <class RawType>
	inline request<RawType> operator>(RawType& m) {
		return operator>( std::move(msg_impl<RawType>(m)) );
	}

	// Returns the rank of this endpoit
	inline const int& rank() const { return m_rank; }
};

} // end mpi namespace 

#include "detail/comm.h"

namespace mpi {

// Send a generic message to this endpoint (synchronously)
template <class MsgType>
inline endpoint& endpoint::send(msg_impl<MsgType>&& m) {
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

// Receive from this endpoing (asynchronously)
template <class MsgType>
inline request<MsgType> endpoint::operator>(msg_impl<MsgType>&& m) {
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

} // end mpi namespace 


#include "detail/status.h"

namespace mpi {

template <class RawType>
inline status endpoint::operator>>(RawType& m) {
	return operator>>( msg_impl<RawType>( m ) );
}

template <class MsgType>
inline status endpoint::operator>>(msg_impl<MsgType>&& m) {
	status::mpi_status_ptr stat( new MPI_Status );
	MPI_Datatype dt = m.type();
	if(MPI_Recv( const_cast<void*>(static_cast<const void*>(m.addr())), 
				 static_cast<int>(m.size()), dt,
				 m_rank, 
				 m.tag(), 
				 m_comm, 
				 stat.get()
			   ) == MPI_SUCCESS ) {
		return status(m_comm, std::move(stat), dt);
	}
	std::ostringstream ss;
	ss << "ERROR in MPI rank '" << comm::world.rank()
	   << "': Failed to receive message from destination rank '"
	   << m_rank << "'";
	throw comm_error( ss.str() );
}

} // end mpi namespace 
