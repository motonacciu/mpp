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

#include "detail/status.h"

namespace mpi {

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
			status::mpi_status_ptr stat(new MPI_Status);
			// wait to receive the message
			MPI_Wait(&m_req, stat.get());
			done = 1;
			m_status.reset( new status(m_comm, std::move(stat), m_msg.type()) );
		}
		return m_msg.get();
	}

	inline status getStatus() {
		if( isDone() ) { return *m_status; }
		throw "not done";
	}

	inline bool isDone() {
		if ( !done ) {
			status::mpi_status_ptr stat(new MPI_Status);
			MPI_Test(&m_req, &done, stat.get());
			if ( done ) {
				m_status.reset( new status(m_comm, std::move(stat), m_msg.type()) );
			}
		}
		return done;
	}
};

} // end mpi namespace 
