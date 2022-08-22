using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour
{
    public Vector3 position;
    public Vector3 velocity;
 
    private void Start()
    {
        position = transform.position;
        velocity = new Vector3(0, 0, 0);
    }




}
