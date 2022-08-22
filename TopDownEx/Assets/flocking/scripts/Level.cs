using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Level : MonoBehaviour
{
    public Transform memberPrefab;
    public Transform enemyPrefab;
    public int numberOfMembers;
    public int numberOfEnemies;
    public List<Member> members;
    public List<Enemy> enemies;
    public float bounds;
    public float spawnRadius;


    // Start is called before the first frame update
    void Start()
    {
        members = new List<Member>();
        enemies = new List<Enemy>();

        Spawn(memberPrefab, numberOfMembers);
        Spawn(enemyPrefab, numberOfEnemies);

        members.AddRange(FindObjectsOfType<Member>());
        enemies.AddRange(FindObjectsOfType<Enemy>());

    }

    void Spawn(Transform prefab, int count) // Tager en prefabs transform og antallet af prefabbens members
    {
        for (int i = 0; i < count; i++)
        {
            Vector3 spawnLoc = new Vector3(Random.Range(-spawnRadius, spawnRadius),
                                           Random.Range(-spawnRadius, spawnRadius),
                                           0);
            Instantiate(prefab, spawnLoc, Quaternion.identity);
        }
    }

    public List<Member> GetNeighbors(Member member, float radius)
    {
        List<Member> neighborsFound = new List<Member>();

        foreach (var otherMember in members)
        {
            if (otherMember == member)
                continue;
            if (Vector3.Distance(member.position, otherMember.position) <= radius) 
            { 
            neighborsFound.Add(otherMember); 
            }
        }

        return neighborsFound;
    }

}
