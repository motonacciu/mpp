#include <gtest/gtest.h>

#include <mpp.h>
#include <iostream>
#include <cmath>

using namespace mpi;

TEST(SendRecv, Scalar) {
	if(comm::world.rank() == 0) {
		comm::world(1) << 4.2;
		int val;
		auto s = comm::world(1) >> val;
		EXPECT_EQ(4, val);
		EXPECT_EQ( 1, s.source().rank() );
		EXPECT_EQ( 0, s.tag() );
	} else if (comm::world.rank() == 1) {
		double val;
		auto s = comm::world(0) >> val;
		EXPECT_EQ(4.2,val);
		EXPECT_EQ( 0, s.source().rank() );
		EXPECT_EQ(0, s.tag());
		comm::world(0) << static_cast<int>(floor(val));
	}
}

TEST(SendRecv, Array) {
	if(comm::world.rank() == 0) {
		comm::world(1) << std::vector<int>( {2, 4, 6, 8} );
	} else if (comm::world.rank() == 1) {
		std::vector<int> vec(4);
		comm::world(0) >> vec;
		EXPECT_EQ( static_cast<size_t>(4), vec.size() );
		std::vector<int> res( {2, 4, 6, 8} );
		// check whether res == vec
		EXPECT_TRUE( std::equal(vec.begin(), vec.end(), res.begin(), std::equal_to<int>()) );
	}
}

TEST(SendRecv, Future) {
	if ( comm::world.rank() == 0 ) {
		comm::world(1) << 100;
	} else if(comm::world.rank() == 1) {
		int k;
		request<int> r = comm::world(0) > k;
		r.get();
		EXPECT_EQ(100, k);
	}
}

TEST(SendRecv, Tags) {

	if ( comm::world.rank() == 0 ) {
		comm::world(1) << msg<const int>(100, 11);
		comm::world(1) << msg<const int>(101, 0);
	} else if(comm::world.rank() == 1) {
		int k;
		comm::world(0) >> msg(k,0);
		EXPECT_EQ(101, k);
		comm::world(0) >> msg(k,11);
		EXPECT_EQ(100, k);
	}
}

TEST(SendRecv, PingPong) {
	int p=0;
	if(comm::world.rank() == 0) {
		// start the ping
		comm::world(1) << p;
	}

	while ( p <= 10 ) {
		auto ep = (comm::world(mpi::any) >> p ).source();
		ep << p+1;
		EXPECT_TRUE(comm::world.rank()==0?p%2!=0:p%2==0);
	}
}

TEST(SendRecv, Lists) {

	if ( comm::world.rank() == 0 ) {
		std::list<int> l = {1,2,3,4,5};
		comm::world(1) << l;
	} else if(comm::world.rank() == 1) {
		std::vector<int> l(5);
		comm::world(0) >> l;
		EXPECT_EQ(static_cast<size_t>(5), l.size());
		EXPECT_EQ(1, l[0]);
		EXPECT_EQ(2, l[1]);
		EXPECT_EQ(3, l[2]);
		EXPECT_EQ(4, l[3]);
		EXPECT_EQ(5, l[4]);
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	// Disables elapsed time by default.
	::testing::GTEST_FLAG(print_time) = false;

	// This allows the user to override the flag on the command line.
	::testing::InitGoogleTest(&argc, argv);

	size_t errcode = RUN_ALL_TESTS();
	MPI_Finalize();
	return errcode;
}
