// ============================================================================
//
//					 	 	MPP: An MPI CPP Interface
//
// 							  Author: Simone Pellegrini
// 		   					    Date:   23 Apr. 2011
//
// ============================================================================

#pragma once

#include <mpi.h>
#include <vector>
#include <list>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <array>

namespace mpi {

class comm;
class endpoint;

//*****************************************************************************
// 									MPI Type Traits
//*****************************************************************************
template <class T>
struct mpi_type_traits {
	static inline MPI_Datatype get_type(const T& raw);
	static inline size_t get_size(const T& raw) { return 1; }
	static inline const T* get_addr(const T& raw) { return &raw; }
};

// primitive type traits
template<>
inline MPI_Datatype mpi_type_traits<double>::get_type(const double&) {
	return  MPI_DOUBLE;
}

template <>
inline MPI_Datatype mpi_type_traits<int>::get_type(const int&) {
	return MPI_INT;
}

template <>
inline MPI_Datatype mpi_type_traits<float>::get_type(const float&) {
	return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_type_traits<long>::get_type(const long&) {
	return MPI_LONG;
}

// ... add missing types here ...

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	std::vector<T> traits
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class T>
struct mpi_type_traits<std::vector<T>> {

	static inline size_t get_size(const std::vector<T>& vec) {
		return vec.size();
	}

	static inline MPI_Datatype  get_type(const std::vector<T>& vec) {
		return  mpi_type_traits<T>::get_type( T() );
	}

	static inline const T* get_addr(const std::vector<T>& vec) {
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
		return  mpi_type_traits<T>::get_addr( vec.front() );
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// msg: represent a single message which can be provided to the <<, <, >>, >
// operations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class MsgTy>
struct msg_impl {
	typedef MsgTy value_type;

	// Builds a msg wrapping v
	msg_impl(MsgTy& v, int tag = 0) : m_data(v), m_tag(tag){ }

	// Returns the address to the first element of the contained data
	inline void* addr() {
		return (void*) mpi_type_traits<MsgTy>::get_addr(m_data);
	}
	inline const void* addr() const {
		return (const void*) mpi_type_traits<MsgTy>::get_addr(m_data);
	}

	inline MsgTy& get() { return m_data; }
	inline const MsgTy& get() const { return m_data; }

	// Returns the dimension of this message
	inline size_t size() const {
		return mpi_type_traits<MsgTy>::get_size(m_data);
	}

	inline MPI_Datatype type() const {
		return mpi_type_traits<MsgTy>::get_type(m_data);
	}

	// getter/setter for m_tag
	inline const int& tag() const { return m_tag; }
	inline int& tag() { return m_tag; }

private:
	MsgTy&  m_data;
	int 	m_tag;
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Specializaton for class MsgTy for const types in this case we don't keep the
// reference to the object passed to the constructor, but we make a copy of it
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <class MsgTy>
struct msg_impl <const MsgTy> {
	typedef const MsgTy value_type;

	// Builds a msg wrapping v
	msg_impl(const MsgTy& v, int tag = 0) : m_data(v), m_tag(tag){ }

	// Returns the enclosed data
	inline const void* addr() const {
		return mpi_type_traits<MsgTy>::get_addr(m_data);
	}

	inline const MsgTy& get() const { return m_data; }

	// Returns the dimension of this message
	inline size_t size() const {
		return mpi_type_traits<MsgTy>::get_size(m_data);
	}

	inline MPI_Datatype type() const {
		return mpi_type_traits<MsgTy>::get_type(m_data);
	}

	// getter/setter for m_tag
	inline const int& tag() const { return m_tag; }
	inline int& tag() { return m_tag; }

private:
	const MsgTy m_data;
	int 		m_tag;
};

template <class T>
msg_impl<T> msg(T& raw, int tag=0) { return msg_impl<T>(raw, tag); }

// Expection which is thrown every time a communication fails
struct comm_error : public std::logic_error {
	comm_error(const std::string& msg) :
		std::logic_error(msg) { }
};

class status;


template <class T>
class request;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// comm: is the abstraction of the MPI_Comm class
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class comm{
	MPI_Comm comm_m;
	int comm_size;

	comm(MPI_Comm comm): comm_m(comm), comm_size(-1) { }

	// Check whether MPI_Init has been called
	void check_init() {
		int flag;
		MPI_Initialized(&flag);
		assert(flag != 0 &&
			"FATAL: MPI environment not initialized (MPI_Init not called)");
	}
public:
	// MPI_COMM_WORLD
	static comm world;

	inline size_t rank() {
		check_init();
		
		int out_rank;
		MPI_Comm_rank(comm_m, &out_rank);
		return out_rank;
	}

	inline size_t size() {
		check_init();
		
		// Get the size for this communicator
		if(comm_size == -1) {
			MPI_Comm_size(comm_m, &comm_size);
		}
		return comm_size;
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

public:
	endpoint(const int& rank, const MPI_Comm& com):
		m_rank(rank), m_comm(com) { }

	// Send a generic message to this endpoint (synchronously)
	template <class MsgType>
	inline endpoint& operator<<(const msg_impl<MsgType>& m) {
		MPI_Datatype&& dt = m.type();
		if ( MPI_Send( const_cast<void*>(m.addr()), static_cast<int>(m.size()), dt,
						m_rank, m.tag(), m_comm
					 ) == MPI_SUCCESS ) {
			return *this;
		}
		std::ostringstream ss;
		ss << "ERROR in MPI rank '" << comm::world.rank()
		   << "': Failed to send message to destination rank '"
		   << m_rank << "'";
		throw comm_error( ss.str() );
	}

	// Send a generic raw type to this endpoint (synchronously)
	template <class RawType>
	inline endpoint& operator<<(const RawType& m) {
		return operator<<(msg_impl<const RawType>(m));
	}

	// Receive from this endpoint (synchronously)
	template <class RawType>
	inline status operator>>(RawType& m);

	template <class MsgType>
	inline status operator>>(const msg_impl<MsgType>& m);

	// Receive from this endpoing (asynchronously)
	template <class MsgType>
	inline request<MsgType> operator>(const msg_impl<MsgType>& m) {
		MPI_Request req;
		if( MPI_Irecv( const_cast<void*>(m.addr()), static_cast<int>(m.size()), m.type(),
					   m_rank, m.tag(), m_comm, &req
					 ) != MPI_SUCCESS ) {
			std::ostringstream ss;
			ss << "ERROR in MPI rank '" << comm::world.rank()
			   << "': Failed to receive message from destination rank '"
			   << m_rank << "'";
			throw comm_error( ss.str() );
		}
		return request<MsgType>(m_comm, req, m);
	}

	// Receive from this endpoing (asynchronously
	template <class RawType>
	inline request<RawType> operator>(RawType& m) {
		return operator>( msg_impl<RawType>( m ) );
	}

	// Returns the rank of this endpoit
	inline const int& rank() const { return m_rank; }
};

endpoint comm::operator()(const int& rank_id) {
	return endpoint(rank_id, comm_m);
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
inline status endpoint::operator>>(const msg_impl<MsgType>& m) {
	MPI_Status s;
	if(MPI_Recv( const_cast<void*>(m.addr()), static_cast<int>(m.size()), m.type(),
				 m_rank, m.tag(), m_comm, &s
			   ) == MPI_SUCCESS ) {
		return status(m_comm, s, m.type());
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
	const msg_impl<T>			m_msg;
	std::shared_ptr<status> 	m_status;
	int 		 				done;

public:
	request(MPI_Comm const& com, MPI_Request req, const msg_impl<T>& msg):
		m_comm(com), m_req(req), m_msg(msg), done(0) { }

	void cancel();

	const T& get() {
		if ( !done ) {
			MPI_Status stat;
			// wait to receive the message
			MPI_Wait(&m_req, &stat);
			done = 1;
			m_status = std::make_shared<status>( m_comm, stat, m_msg.type() );
		}
		return m_msg.get();
	}

	status getStatus() {
		if( isDone() ) {
			return *m_status;
		}
		throw "not done";
	}

	bool isDone() {
		if ( !done ) {
			MPI_Status stat;
			MPI_Test(&m_req, &done, &stat);
			if ( done ) {
				m_status = std::make_shared<status>( m_comm, stat, m_msg.type() );
			}
		}
		return done;
	}
};

const int any = MPI_ANY_SOURCE;

void init(int argc = 0, char* argv[] = NULL){ MPI_Init(&argc, &argv); }
void finalize(){ MPI_Finalize(); }

} // end mpi namespace

