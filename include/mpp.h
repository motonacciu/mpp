/******************************************************************************
 *
 *                          MPP: An MPI CPP Interface
 *
 *                  Copyright (C) 2011-2012  Simone Pellegrini
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 ******************************************************************************/

#pragma once

#include "detail/comm.h"
#include "detail/endpoint.h"
#include "detail/message.h"
#include "detail/status.h"
#include "detail/request.h"

namespace mpi {

const int any = MPI_ANY_SOURCE;

inline void init(int argc = 0, char* argv[] = NULL, int required = 0, int* provided = 0) {
  if(provided == 0) {
    MPI_Init(&argc, &argv);
  } else {
    MPI_Init_thread( &argc, &argv, required, provided );
  }
}

inline void finalize(){ MPI_Finalize(); }

} // end mpi namespace

