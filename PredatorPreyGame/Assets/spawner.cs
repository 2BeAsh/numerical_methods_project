using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class spawner : MonoBehaviour
{
    public GameObject objectToSpawn;
    public float spawnRadius;
    public int numberToSpawn;


    private void Start()
    {
        CreateObjects(numberToSpawn);
    }


    private void CreateObjects(int objectNum)
    {
        for (int i = 0; i < objectNum; i++)
        {
            Vector3 spawnLoc = new Vector3(Random.Range(-spawnRadius, spawnRadius),
                                           Random.Range(-spawnRadius, spawnRadius),
                                           0);
            GameObject objectClone = Instantiate(objectToSpawn, spawnLoc, Quaternion.identity);
        }

    }

}

