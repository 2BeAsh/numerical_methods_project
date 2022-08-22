using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class flockingBehaviour : MonoBehaviour
{
    [SerializeField] public GameObject preyPrefab;
    public int numPrey = 20;
    public GameObject[] allPrey { get; set; }
    public float spawnRadius;

    private void Start()
    {
        GenerateUnits();
    }

    private void GenerateUnits()
    {
        allPrey = new GameObject[numPrey];
        for (int i = 0; i< numPrey; i++)
        {
            Vector2 spawnLoc = new Vector2(Random.Range(-spawnRadius, spawnRadius),
                                           Random.Range(-spawnRadius, spawnRadius));
            allPrey[i] = Instantiate(preyPrefab, spawnLoc, Quaternion.identity);
        }
    }





}
