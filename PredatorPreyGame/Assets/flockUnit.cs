using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class flockUnit : MonoBehaviour
{

    private List<flockUnit> Neighbours = new List<flockUnit>();
    private flockingBehaviour assignedFlock;

    public void AssignFlock(flockingBehaviour flock)
    {

    }

public void MoveUnit()
    {
        FindNeighbours();
    }


    private void FindNeighbours()
    {
        Neighbours.Clear();
        //CanvasRenderer allUnits = 
    }
}
