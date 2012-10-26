from pyla.accessors import *
import unittest

class TestColumnAccessor(unittest.TestCase):

    def setUp(self):
        self.a = [[1,2,3,4],
                  [5,6,7,8],
                  [0,1,0,1]]
        self.a0 = [row[:] for row in self.a]

    def test_element_access(self):
        row0 = ColumnView( self.a, 0 )
        
        self.assertEqual( row0[0], 1 )
        self.assertEqual( row0[2], 0 )
        self.assertEqual( row0[-1], 0 )
        self.assertEqual( len(row0), 3 )

    def test_iteration(self):
        row0 = ColumnView( self.a, 0 )
        self.assertEqual( list(row0), [1,5,0] )
        
    def test_slice_access(self):
        row0 = ColumnView( self.a, 0 )
    
        self.assertEqual( list(row0[:]), [1,5,0] )
        self.assertEqual( list(row0[::-1]), [0,5,1] )
        self.assertEqual( list(row0[1:]), [5,0] )
        self.assertEqual( list(row0[1:2]), [5] )
        self.assertEqual( list(row0[1:3]), [5,0] )
        self.assertEqual( list(row0[1:3]), [5,0] )
                          
        
    def test_element_update(self):
        row = ColumnView( self.a, 1 )
        row[0] = -1
        self.assertEqual( row[0], -1 )
        self.assertEqual( self.a,
                          [[1,-1,3,4],
                           [5,6,7,8],
                           [0,1,0,1]] )

    def test_slice_update(self):
        row = ColumnView( self.a, 1 )

        row[:] = [5,5,5]
        self.assertEqual( list(row), [5,5,5] )
        self.assertEqual( self.a,
                          [[1,5,3,4],
                           [5,5,7,8],
                           [0,5,0,1]] )

        row[::-1] = [1,2,3]
        self.assertEqual( list(row), [3,2,1] )
        
        
    def test_slice_update(self):
        row = ColumnView( self.a, 1 )

        row[:] = [5,5,5]
        self.assertEqual( list(row), [5,5,5] )
        self.assertEqual( self.a,
                          [[1,5,3,4],
                           [5,5,7,8],
                           [0,5,0,1]] )
        


class TestTransposedAccessor(unittest.TestCase):
    def setUp(self):
        self.a = [[1,2,3,4],
                  [5,6,7,8],
                  [0,1,0,1]]
        self.at = TransposedView(self.a)

    def test_size(self):
        self.assertEqual(len(self.at), 4)
        self.assertEqual(len(self.at[0]), 3)

    def test_element_access(self):
        for i in range(3):
            for j in range(4):
                self.assertEqual(self.a [i][j],
                                 self.at[j][i])
    def test_element_update(self):
        a = self.a
        at = self.at

        at[0][0] = -1
        self.assertEqual( a[0][0], -1 )

        at[1][0] = -1
        self.assertEqual( a[0][1], -1 )
        at[3][1] = -1
        self.assertEqual( a[1][3], -1 )

        
    def test_row_update(self):
        a = self.a
        at = self.at
        
        at[0] = [5,5,5]

        self.assertEqual( a,
                          [[5,2,3,4],
                           [5,6,7,8],
                           [5,1,0,1]] )

        at[-1] = [6,6,6]
        self.assertEqual( a,
                          [[5,2,3,6],
                           [5,6,7,6],
                           [5,1,0,6]] )


    def test_row_range_update(self):
        a = self.a
        at = self.at
        
        at[0:2] = [[1,1,1],[2,2,2]]
        
        self.assertEqual( a,
                          [[1,2,3,4],
                           [1,2,7,8],
                           [1,2,0,1]] )
